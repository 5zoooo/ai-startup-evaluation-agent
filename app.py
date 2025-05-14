# app.py
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone

# 에이전트 불러오기
from agents.summary_agent import get_summary_agent
from agents.market_agent import get_market_agent
from agents.risk_agent import get_risk_agent
from agents.decision_agent import get_decision_agent

# 1. 환경변수 로드 및 설정
load_dotenv()
query = "보이저엑스 관련 정보"

# 2. Pinecone 초기화
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "investment-analysis-final"

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
)

# 3. 문서 검색 -- 수정 필요
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.get_relevant_documents(query)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# 4. 에이전트 체인 실행 -- 수정 필요
summary = get_summary_agent().run({"context": context})
print("📝 기업 요약:\n", summary)

market = get_market_agent().run({"context": context})
print("\n📊 시장성 평가:\n", market)

risk = get_risk_agent().run({"context": context})
print("\n⚠️ 리스크 분석:\n", risk)

decision = get_decision_agent().run({
    "summary": summary,
    "market": market,
    "risk": risk
})
print("\n📌 투자 판단:\n", decision)
