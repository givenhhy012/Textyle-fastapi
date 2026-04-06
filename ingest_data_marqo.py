
import sys
import psycopg2
from pgvector.psycopg2 import register_vector
from PIL import Image
import torch
import open_clip
import logging
from datasets import load_dataset
import base64
import io


# 현재 테스트 용으로 huggingface api이용해서 이미지 벡터 추출.
# image_url은 임시적으로 이미지 원본을 base64 형태로 변환 후 DB에 적재.
# 이후 실제 이미지를 S3에 저장할 경우, 실제 image 주소를 활용할 것.


# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 데이터베이스 연결 정보 ---
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "1234"
MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"

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
    """Marqo-FashionSigLIP 모델을 로드합니다."""
    logging.info(f"'{MODEL_NAME}' 모델 로드 중...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model.eval()
    logging.info("모델 로드 완료.")
    return model, preprocess


def encode_images_batch(model, preprocess, pil_images: list) -> list:
    """PIL 이미지 리스트를 배치로 768차원 벡터로 변환 후 정규화"""
    image_tensors = torch.stack([preprocess(img) for img in pil_images])
    with torch.no_grad():
        features = model.encode_image(image_tensors)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def get_or_create_category(cur, category_name, category_cache):
    """
    카테고리 이름을 파싱하여 DB에 저장하고 id를 반환합니다.
    인메모리 캐시를 통해 중복 INSERT를 방지합니다.

    DeepFashion 카테고리 형식: "MEN-Denim" 또는 "WOMEN-Blouses_Shirts"
      -> 대분류(level 1): "MEN" / "WOMEN"
      -> 소분류(level 2): "Denim" / "Blouses Shirts"

    단일 단어 카테고리는 level 1 단독 처리.
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
        # 단일 카테고리 (level 1)
        single_name = category_name.replace('_', ' ').strip()
        cur.execute(
            "INSERT INTO categories (name, level) VALUES (%s, 1) RETURNING id",
            (single_name,)
        )
        cat_id = cur.fetchone()[0]
        category_cache[category_name] = cat_id
        return cat_id


def save_batch_to_db(conn, metadata, embeddings):
    """배치 단위로 DB에 저장 (categories → products → product_images 순서)"""
    with conn.cursor() as cur:
        for meta, emb in zip(metadata, embeddings):
            # 1. 카테고리 삽입 (캐시 활용)
            category_id = get_or_create_category(
                cur, meta.get("category_name"), meta["category_cache"]
            )

            # 2. 상품 삽입
            cur.execute(
                """
                INSERT INTO products (name, category_id, source)
                VALUES (%s, %s, 'DEEPFASHION')
                RETURNING id
                """,
                (meta["name"], category_id)
            )
            product_id = cur.fetchone()[0]

            # 3. 이미지 및 벡터 삽입
            cur.execute(
                """
                INSERT INTO product_images (product_id, image_url, is_main, embedding)
                VALUES (%s, %s, TRUE, %s)
                """,
                (product_id, meta["image_url"], emb.tolist())
            )


def process_streaming(limit=1000):
    """HuggingFace 데이터셋을 스트리밍하여 DB에 적재"""
    conn = get_db_connection()
    model, preprocess = load_model()

    # 카테고리 인메모리 캐시: {category_name_string: db_id}
    category_cache = {}

    try:
        # --- 기존 데이터 전체 삭제 및 ID 초기화 ---
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

        logging.info("벡터 추출 및 DB 적재 시작...")

        for item in dataset:
            if count >= limit:
                break

            # 이미지 컬럼명 자동 감지
            img_key = 'image' if 'image' in item else ('img' if 'img' in item else None)
            if img_key is None:
                logging.error("이미지 컬럼을 찾을 수 없습니다. 데이터셋 구조를 확인하세요.")
                break

            img = item[img_key].convert("RGB")

            # 이미지 → Base64 변환 (임시 저장 방식, 추후 S3 URL로 교체 예정)
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
                embeddings = encode_images_batch(model, preprocess, batch_images)
                save_batch_to_db(conn, batch_metadata, embeddings)
                conn.commit()

                count += len(batch_images)
                logging.info(f"적재 완료: {count}건 (카테고리 캐시: {len(category_cache)}개)")

                batch_images = []
                batch_metadata = []

        # 마지막 남은 배치 처리
        if batch_images:
            embeddings = encode_images_batch(model, preprocess, batch_images)
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
    process_streaming(limit=1000)
