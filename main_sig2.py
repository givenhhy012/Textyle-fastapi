
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from pgvector.psycopg2 import register_vector
from PIL import Image
import io
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
import logging
from contextlib import asynccontextmanager

# python main_sig2.py 이후 브라우저에서 http://localhost:8000/docs 접속.
# SigLIP 2 모델을 사용한 벡터 검색 서버 (768차원)

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 데이터베이스 연결 정보 ---
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "1234"

# SigLIP 2 모델: 768차원 벡터 출력, 다국어(한국어) 지원
MODEL_NAME = "google/siglip2-base-patch16-256"

# --- 전역 변수 (모델) ---
model = None
processor = None

# composed 검색 시 이미지/텍스트 임베딩 결합 비율
alpha = 0.5
beta = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 정리"""
    global model, processor
    logging.info(f"'{MODEL_NAME}' 모델을 로드하는 중...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    logging.info("모델 로드 완료. (SigLIP 2, 768차원)")
    yield
    logging.info("서버를 종료합니다.")


app = FastAPI(title="Textyle Vector Search API (SigLIP 2)", lifespan=lifespan)

# CORS 설정 (프론트엔드 연결 대비)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    """PostgreSQL 데이터베이스 연결"""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    register_vector(conn)
    return conn


def encode_image(pil_image: Image.Image) -> np.ndarray:
    """PIL 이미지를 768차원 벡터로 변환 후 정규화"""
    inputs = processor(images=[pil_image], return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()


def encode_text(text: str) -> np.ndarray:
    """텍스트를 768차원 벡터로 변환 후 정규화"""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()


@app.get("/")
async def root():
    return {"message": "Textyle Vector Search API (SigLIP 2) is running"}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    """이미지를 업로드하여 유사한 상품 검색"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        logging.info("이미지 임베딩 생성 중...")
        embedding = encode_image(image)

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        p.id,
                        p.name,
                        pi.image_url,
                        1 - (pi.embedding <=> %s::vector) AS similarity
                    FROM product_images pi
                    JOIN products p ON pi.product_id = p.id
                    ORDER BY pi.embedding <=> %s::vector
                    LIMIT 5
                """
                cur.execute(query, (embedding.tolist(), embedding.tolist()))
                results = cur.fetchall()

                return {
                    "results": [
                        {
                            "product_id": row[0],
                            "name": row[1],
                            "image_url": row[2],
                            "similarity": float(row[3])
                        }
                        for row in results
                    ]
                }
        finally:
            conn.close()

    except Exception as e:
        logging.error(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/composed")
async def search_composed(text: str = Form(...), file: UploadFile = File(...)):
    """이미지와 텍스트를 결합하여 유사한 상품 검색 (CIR)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        logging.info(f"결합 검색 시작: 텍스트='{text}'")
        img_emb = encode_image(image)
        txt_emb = encode_text(text)

        # 이미지 벡터에 텍스트로 표현된 수정 의도를 결합 후 재정규화
        combined_emb = (alpha * img_emb) + (beta * txt_emb)
        combined_emb = combined_emb / np.linalg.norm(combined_emb)

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        p.id,
                        p.name,
                        pi.image_url,
                        1 - (pi.embedding <=> %s::vector) AS similarity
                    FROM product_images pi
                    JOIN products p ON pi.product_id = p.id
                    ORDER BY pi.embedding <=> %s::vector
                    LIMIT 5
                """
                cur.execute(query, (combined_emb.tolist(), combined_emb.tolist()))
                results = cur.fetchall()

                return {
                    "query_text": text,
                    "results": [
                        {
                            "product_id": row[0],
                            "name": row[1],
                            "image_url": row[2],
                            "similarity": float(row[3])
                        }
                        for row in results
                    ]
                }
        finally:
            conn.close()

    except Exception as e:
        logging.error(f"결합 검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
