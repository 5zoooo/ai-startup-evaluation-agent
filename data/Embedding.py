import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. 환경 변수 로딩
load_dotenv()

# 2. Pinecone 연결
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "investment-analysis-final"

# 3. 인덱스가 없으면 생성
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI 임베딩 차원 수
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # 👉 실제 Pinecone 인덱스 지역
        )
    )

# 4. 인덱스 연결
index = pc.Index(index_name)

# 5. OpenAI 임베딩 모델 준비
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 6. 문서 분할 전략 정의
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", " ", ""]
)

# 7. PDF 파일 리스트
pdf_files = [
    ("AI News.pdf", "ai_news"),
    ("Startup Evaluation Metrics.pdf", "startup_metrics"),
    ("VoyagerX Introduction.pdf", "voyagerx_intro"),
    ("Video Stt Script.pdf", "video_stt_script")  # ✅ 추가된 PDF
]

all_chunks = []

# 8. 각 PDF 로딩, 분할, 메타데이터 부여
for filename, doc_id in pdf_files:
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    for i, doc in enumerate(pages):
        doc.metadata["source"] = doc_id
        doc.metadata["page"] = i + 1

    chunks = text_splitter.split_documents(pages)
    all_chunks.extend(chunks)

# 9. Pinecone에 업로드
vectorstore = PineconeVectorStore.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    index_name=index_name
)
