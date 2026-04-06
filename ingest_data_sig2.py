
import sys
import psycopg2
from pgvector.psycopg2 import register_vector
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
import logging
from datasets import load_dataset
import base64
import io


# SigLIP 2 모델을 사용한 데이터 적재 스크립트 (768차원)
# 기존 ingest_data.py(FashionCLIP, 512차원)와 동일한 로직이나 모델만 다름.
# 테스트 후 모델 확정 시 하나로 통합 예정.


# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 데이터베이스 연결 정보 ---
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "1234"

# SigLIP 2 모델: 768차원 벡터 출력
MODEL_NAME = "google/siglip2-base-patch16-256"

# --- 데이터셋 설정 ---
DATASET_PATH = "Marqo/deepfashion-inshop"


def get_db_connection():
    """PostgreSQL 데이터베이스에 연결합니다."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        register_vector(conn)
        return conn
    except Exception as e:
        logging.error(f"DB 연결 실패: {e}")
        sys.exit(1)


def load_model():
    """SigLIP 2 모델을 로드합니다."""
    logging.info(f"'{MODEL_NAME}' 모델 로드 중...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    logging.info("모델 로드 완료. (SigLIP 2, 768차원)")
    return model, processor


def encode_images_batch(model, processor, pil_images: list):
    """PIL 이미지 리스트를 배치로 768차원 벡터로 변환 후 정규화"""
    inputs = processor(images=pil_images, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def get_or_create_category(cur, category_name, category_cache):
    """
    카테고리 이름을 파싱하여 DB에 저장하고 id를 반환합니다.
    인메모리 캐시를 통해 중복 INSERT를 방지합니다.

    DeepFashion 카테고리 형식: "MEN-Denim" 또는 "WOMEN-Blouses_Shirts"
      -> 대분류(level 1): "MEN" / "WOMEN"
      -> 소분류(level 2): "Denim" / "Blouses Shirts"
    """
    if not category_name:
        return None

    if category_name in category_cache:
        return category_cache[category_name]

    parts = category_name.split('-', 1)

    if len(parts) == 2:
        parent_name = parts[0].strip()
        child_name = parts[1].replace('_', ' ').strip()

        # 대분류 조회 또는 삽입
        if parent_name not in category_cache:
            cur.execute(
                "INSERT INTO categories (name, level) VALUES (%s, 1) RETURNING id",
                (parent_name,)
            )
            category_cache[parent_name] = cur.fetchone()[0]

        parent_id = category_cache[parent_name]

        # 소분류 삽입
        cur.execute(
            "INSERT INTO categories (name, parent_id, level) VALUES (%s, %s, 2) RETURNING id",
            (child_name, parent_id)
        )
        child_id = cur.fetchone()[0]
        category_cache[category_name] = child_id
        return child_id

    else:
        single_name = category_name.replace('_', ' ').strip()
        cur.execute(
            "INSERT INTO categories (name, level) VALUES (%s, 1) RETURNING id",
            (single_name,)
        )
        cat_id = cur.fetchone()[0]
        category_cache[category_name] = cat_id
        return cat_id


def save_batch_to_db(conn, metadata, embeddings):
    """배치 단위로 DB에 저장 (categories → products → product_images)"""
    with conn.cursor() as cur:
        for meta, emb in zip(metadata, embeddings):
            category_id = get_or_create_category(
                cur, meta.get("category_name"), meta["category_cache"]
            )

            cur.execute(
                """
                INSERT INTO products (name, category_id, source)
                VALUES (%s, %s, 'DEEPFASHION')
                RETURNING id
                """,
                (meta["name"], category_id)
            )
            product_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO product_images (product_id, image_url, is_main, embedding)
                VALUES (%s, %s, TRUE, %s)
                """,
                (product_id, meta["image_url"], emb.tolist())
            )


def process_streaming(limit=100):
    """HuggingFace 데이터셋을 스트리밍하여 DB에 적재"""
    conn = get_db_connection()
    model, processor = load_model()

    category_cache = {}

    try:
        with conn.cursor() as cur:
            logging.info("기존 데이터를 삭제하고 ID를 초기화합니다...")
            cur.execute("TRUNCATE TABLE product_attributes RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE product_images    RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE products          RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE categories        RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE attributes        RESTART IDENTITY CASCADE;")
            conn.commit()

        logging.info(f"데이터셋 '{DATASET_PATH}' 로드 중 (Streaming Mode)...")
        dataset = load_dataset(DATASET_PATH, split="data", streaming=True, trust_remote_code=True)

        batch_size = 32
        batch_images = []
        batch_metadata = []
        count = 0

        logging.info("벡터 추출 및 DB 적재 시작 (SigLIP 2)...")

        for item in dataset:
            if count >= limit:
                break

            img_key = 'image' if 'image' in item else ('img' if 'img' in item else None)
            if img_key is None:
                logging.error("이미지 컬럼을 찾을 수 없습니다.")
                break

            img = item[img_key].convert("RGB")

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_data_url = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

            item_id = str(item.get('item_ID', item.get('id', f"hf_{count}")))
            category_name = item.get('category_name', None)

            batch_images.append(img)
            batch_metadata.append({
                "name": item_id,
                "image_url": img_data_url,
                "category_name": category_name,
                "category_cache": category_cache,
            })

            if len(batch_images) == batch_size:
                embeddings = encode_images_batch(model, processor, batch_images)
                save_batch_to_db(conn, batch_metadata, embeddings)
                conn.commit()

                count += len(batch_images)
                logging.info(f"적재 완료: {count}건 (카테고리 캐시: {len(category_cache)}개)")

                batch_images = []
                batch_metadata = []

        if batch_images:
            embeddings = encode_images_batch(model, processor, batch_images)
            save_batch_to_db(conn, batch_metadata, embeddings)
            conn.commit()
            count += len(batch_images)
            logging.info(f"최종 적재 완료: {count}건 (카테고리 캐시: {len(category_cache)}개)")

    except Exception as e:
        logging.error(f"처리 중 오류 발생: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    process_streaming(limit=100)
