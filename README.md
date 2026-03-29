# Textyle Vector Search 프로젝트 세팅 가이드

  현재 HuggingFace DeepFashion 데이터셋을 스트리밍하여 PostgreSQL + pgvector에 적재하고, FastAPI를 통해     
  이미지검색 기능

  ### 1. 사전 준비 사항
  시작하기 전에 다음 도구들이 설치되어 있어야 함.
   * Python 3.9+
   * Docker Desktop (데이터베이스 실행용)
   * Git

  ---

  ### 2. 환경 구축 순서

  1) 저장소 클론 및 라이브러리 설치

    1 # 저장소 클론
    2 git clone <저장소-URL>
    3 cd textyle-vector
    4
    5 # 가상환경 생성 (권장)
    6 python -m venv venv
    7 source venv/Scripts/activate  # Windows
    8 # source venv/bin/activate    # Mac/Linux
   
  필수 라이브러리 설치
   * pip install -r requirements.txt
   * pip install fastapi uvicorn python-multipart datasets

  2) Docker를 이용한 DB 서버 실행
  pgvector가 포함된 PostgreSQL 이미지를 사용하여 DB 서버를 띄우기.

   * pgvector 전용 이미지를 사용하여 컨테이너 실행
   * 포트 5433, 비밀번호 1234 (소스코드 설정값과 동일)
   * docker run --name textyle-db -e POSTGRES_PASSWORD=1234 -p 5433:5432 -d ankane/pgvector

  3) 데이터베이스 스키마 생성
   * DB 관리 도구(DBeaver, pgAdmin 등)를 통해 localhost:5433에 접속. (ID: postgres, PW: 1234)
   * database_update.txt 파일에 있는 쿼리문을 복사하여 실행.
       * 주의: 최상단의 CREATE EXTENSION IF NOT EXISTS vector;가 반드시 실행되어야 함.

  ---

  ### 3. 데이터 적재 및 서버 실행

  1) 초기 데이터 적재 (HuggingFace Streaming)
  : HuggingFace에서 데이터를 실시간으로 읽어와 벡터를 생성하고 DB에 저장. (이미지는 Base64로 저장되어 별도 서버가
  필요 없음.)
   * python ingest_data.py
     * 기본적으로 100~128건의 테스트 데이터를 적재하도록 설정되어 있음.
     * 적재가 완료되면 DB의 products와 product_images 테이블에서 데이터를 확인가능.

  2) API 서버 실행
   * python main.py
     * 서버가 시작되면 모델(FashionCLIP)을 로드하는 데 약간의 시간이 소요됩니다.
     * Swagger UI 접속: http://localhost:8000/docs (http://localhost:8000/docs)

  ---

  ### 4. API 사용법
  이미지 검색 (/search/image)
   1. Swagger UI의 /search/image 엔드포인트에서 Try it out을 클릭.
   2. file 항목에 검색하고 싶은 의류 이미지(sample_image.png 등)를 업로드.
   3. Execute를 누르면 유사한 Top 5 상품 정보와 Base64 이미지 데이터가 터미널 로그로 반환.

  ---

  📁 주요 파일 설명
   * ingest_data.py: HuggingFace 데이터를 스트리밍하여 DB에 적재하는 스크립트.
   * main.py: FastAPI 기반 이미지 검색 API 서버.
   * database.txt: PostgreSQL 테이블 생성 및 인덱스 설정 DDL.
   * requirements.txt: 프로젝트 의존성 라이브러리 목록.
