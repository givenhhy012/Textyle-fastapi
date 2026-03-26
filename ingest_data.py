
import os
import sys
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from PIL import Image
import torch
from fashion_clip.fashion_clip import FashionCLIP
import logging
from datasets import load_dataset
import base64
import io


# 현재 테스트 용으로 huggingface api이용해서 이미지 벡터 추출
# 128개의 데이터(api에서 순서상 앞쪽 번호가 MEN-denim이라 남자 데님밖에 없을거임) 적재.
# image_url은 임시적으로 이미지원본을 base64형태로 변환후 그걸 db에 적재. 
# 이후 실제 이미지를 크롤링해서 사용할 경우, 실제 image 주소를 활용할거임.
# 


# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 데이터베이스 연결 정보 ---
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "1234"
MODEL_NAME = "fashion-clip"

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
    """FashionCLIP 모델을 로드합니다."""
    fclip = FashionCLIP(MODEL_NAME)
    # Patch for transformers v4+ compatibility
    def patch_clip_model(model):
        for attr_name in ["get_image_features", "get_text_features"]:
            if hasattr(model, attr_name):
                original_func = getattr(model, attr_name)
                def wrapped_func(*args, _original_func=original_func, **kwargs):
                    outputs = _original_func(*args, **kwargs)
                    if hasattr(outputs, "pooler_output"): return outputs.pooler_output
                    if isinstance(outputs, (list, tuple)) and len(outputs) > 1: return outputs[1]
                    return outputs
                setattr(model, attr_name, wrapped_func)
    patch_clip_model(fclip.model)
    return fclip

def process_streaming(limit=1000):
    """HuggingFace 데이터셋을 스트리밍하여 DB에 적재 (이미지는 Base64로 저장)"""
    conn = get_db_connection()
    fclip = load_model()
    
    try:
        # --- 기존 데이터 삭제 및 ID 초기화 ---
        with conn.cursor() as cur:
            logging.info("기존 데이터를 삭제하고 ID를 초기화합니다 (TRUNCATE RESTART IDENTITY)...")
            cur.execute("TRUNCATE TABLE product_images RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE products RESTART IDENTITY CASCADE;")
            conn.commit()

        logging.info(f"데이터셋 '{DATASET_PATH}' 로드 중 (Streaming Mode, Split: data)...")
        dataset = load_dataset(DATASET_PATH, split="data", streaming=True, trust_remote_code=True)
        
        batch_size = 32
        batch_images = []
        batch_metadata = []
        count = 0

        logging.info("벡터 추출 및 DB 적재 시작 (Base64 인코딩 포함)...")
        
        for item in dataset:
            if count >= limit:
                break
            
            # 컬럼명 자동 감지
            img_key = 'image' if 'image' in item else ('img' if 'img' in item else None)
            if img_key is None:
                logging.error(f"이미지 컬럼을 찾을 수 없습니다.")
                break

            # PIL 이미지 추출
            img = item[img_key].convert("RGB")
            
            # --- 이미지를 Base64로 변환 (JPEG 압축 포함) ---
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_data_url = f"data:image/jpeg;base64,{img_base64}"
            
            # 아이템 식별자
            item_id = str(item.get('item_ID', item.get('id', f"hf_{count}")))
            
            batch_images.append(img)
            batch_metadata.append({
                "name": item_id,
                "image_url": img_data_url 
            })
            
            if len(batch_images) == batch_size:
                # 1. 벡터 추출
                embeddings = fclip.encode_images(batch_images, batch_size=batch_size)
                
                # 2. DB 삽입
                save_batch_to_db(conn, batch_metadata, embeddings)
                
                count += len(batch_images)
                logging.info(f"현재 적재 완료: {count}건")
                
                # 초기화
                batch_images = []
                batch_metadata = []

        # 남은 데이터 처리
        if batch_images:
            embeddings = fclip.encode_images(batch_images, batch_size=len(batch_images))
            save_batch_to_db(conn, batch_metadata, embeddings)
            count += len(batch_images)
            logging.info(f"최종 적재 완료: {count}건")

        conn.commit()
    except Exception as e:
        logging.error(f"처리 중 오류 발생: {e}")
        conn.rollback()
    finally:
        conn.close()

def save_batch_to_db(conn, metadata, embeddings):
    """배치 단위로 DB에 저장 (상품 및 이미지)"""
    with conn.cursor() as cur:
        for meta, emb in zip(metadata, embeddings):
            # 1. 상품 삽입 (RETURNING id)
            cur.execute(
                "INSERT INTO products (name, source) VALUES (%s, 'DEEPFASHION') RETURNING id",
                (meta["name"],)
            )
            product_id = cur.fetchone()[0]
            
            # 2. 이미지 및 벡터 삽입
            cur.execute(
                "INSERT INTO product_images (product_id, image_url, embedding) VALUES (%s, %s, %s)",
                (product_id, meta["image_url"], emb.tolist())
            )

if __name__ == "__main__":
    # 테스트를 위해 100건만 먼저 받아봅니다.
    process_streaming(limit=100)