
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from pgvector.psycopg2 import register_vector
from PIL import Image
import io
import torch
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import logging
from contextlib import asynccontextmanager

# python main.py이후 브라우저에서 http://localhost:8000/docs 접속.
# POST /search/image 클릭 후, Try it out => 이미지 파일 첨부 후 execute

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 데이터베이스 연결 정보 ---
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "1234"
MODEL_NAME = "fashion-clip"

# --- 전역 변수 (모델) ---
fclip = None

# combined_emb = (alpha * img_emb) + (beta * txt_emb)
alpha = 0.5
beta = 0.5


def patch_clip_model(model):
    """FashionCLIP library fix: 'BaseModelOutputWithPooling' object has no attribute 'detach'"""
    for attr_name in ["get_image_features", "get_text_features"]:
        if hasattr(model, attr_name):
            original_func = getattr(model, attr_name)
            def wrapped_func(*args, _original_func=original_func, **kwargs):
                outputs = _original_func(*args, **kwargs)
                if hasattr(outputs, "pooler_output"):
                    return outputs.pooler_output
                if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                    return outputs[1]
                return outputs
            setattr(model, attr_name, wrapped_func)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 정리"""
    global fclip
    logging.info(f"'{MODEL_NAME}' 모델을 로드하는 중...")
    fclip = FashionCLIP(MODEL_NAME)
    patch_clip_model(fclip.model)
    logging.info("모델 로드 완료.")
    yield
    logging.info("서버를 종료합니다.")

app = FastAPI(title="Textyle Vector Search API", lifespan=lifespan)

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

@app.get("/")
async def root():
    return {"message": "Textyle Vector Search API is running"}

@app.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    """이미지를 업로드하여 유사한 상품 검색"""
    if not fclip:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 1. 이미지 읽기 및 전처리
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 2. 임베딩 생성
        logging.info("이미지 임베딩 생성 중...")
        embedding = fclip.encode_images([image], batch_size=1)[0]
        
        # 3. DB 검색
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # <=> 연산자: 코사인 거리 (작을수록 유사)
                # %s::vector를 통해 명시적으로 벡터 타입임을 선언합니다.
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
                
                search_results = []
                for row in results:
                    search_results.append({
                        "product_id": row[0],
                        "name": row[1],
                        "image_url": row[2],
                        "similarity": float(row[3])
                    })
                
                return {"results": search_results}
        finally:
            conn.close()

    except Exception as e:
        logging.error(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/composed")
async def search_composed(text: str = Form(...), file: UploadFile = File(...)):
    """이미지와 텍스트를 결합하여 유사한 상품 검색 (CIR)"""
    if not fclip:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 1. 이미지 및 텍스트 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 2. 각각의 임베딩 생성
        logging.info(f"결합 검색 시작: 텍스트='{text}'")
        # FashionCLIP은 이미지와 텍스트를 동일한 512차원 공간으로 임베딩합니다.
        img_emb = fclip.encode_images([image], batch_size=1)[0]
        txt_emb = fclip.encode_text([text], batch_size=1)[0]
        
        # 3. 벡터 결합 (단순 합산 후 정규화)
        # 이미지의 특징에 텍스트로 표현된 '수정 의도'를 더하는 방식입니다.
        combined_emb = (alpha * img_emb) + (beta * txt_emb)
        combined_emb = combined_emb / np.linalg.norm(combined_emb)
        
        # 4. DB 검색
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
                
                search_results = []
                for row in results:
                    search_results.append({
                        "product_id": row[0],
                        "name": row[1],
                        "image_url": row[2],
                        "similarity": float(row[3])
                    })
                
                return {
                    "query_text": text,
                    "results": search_results
                }
        finally:
            conn.close()

    except Exception as e:
        logging.error(f"결합 검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)