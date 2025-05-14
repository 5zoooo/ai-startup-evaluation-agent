from typing import TypedDict

# app.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import StateGraph, END

# 에이전트 임포트
from agents.summary_agent import get_summary_agent
from agents.market_agent import get_market_agent
from agents.risk_agent import get_risk_agent
from agents.decision_agent import get_decision_agent

# ✅ 환경 로딩
load_dotenv()
query = "보이저엑스 관련 정보"

# ✅ Pinecone 벡터 검색 준비
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "investment-analysis-final"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# ✅ 에이전트 체인 초기화
summary_chain = get_summary_agent()
market_chain = get_market_agent()
risk_chain = get_risk_agent()
decision_chain = get_decision_agent()

# ✅ LangGraph 정의
class AgentState(TypedDict, total=False):
    context: str
    summary_result: str
    market_result: str
    risk_result: str
    decision_result: str

builder = StateGraph(AgentState)

# 1. 요약 → context만 사용
builder.add_node("summary", lambda state: {
    "summary_result": summary_chain.run({"context": state["context"]})
})

# 2. 중간 노드: market + risk 동시 처리
builder.add_node("market_and_risk", lambda state: {
    "market_result": market_chain.run({
        "context": state["context"],
        "summary": state["summary_result"]
    }),
    "risk_result": risk_chain.run({"context": state["context"]})
})

# 3. 최종 판단
builder.add_node("decision", lambda state: {
    "decision_result": decision_chain.run({
        "summary": state["summary_result"],
        "market": state["market_result"],
        "risk": state["risk_result"]
    })
})

# ✅ 그래프 흐름 정의
builder.set_entry_point("summary")
builder.add_edge("summary", "market_and_risk")
builder.add_edge("market_and_risk", "decision")
builder.add_edge("decision", END)

# ✅ 실행
graph = builder.compile()
result = graph.invoke({"context": context})

# ✅ 출력
print("📝 요약:\n", result["summary_result"])
print("\n📊 시장성:\n", result["market_result"])
print("\n⚠️ 리스크:\n", result["risk_result"])
print("\n📌 판단:\n", result["decision_result"])
