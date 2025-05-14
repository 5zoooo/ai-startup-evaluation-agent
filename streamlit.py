import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import StateGraph, END
from outputs.report_pdf import save_report_to_pdf

# 에이전트 임포트
from agents.summary_agent import get_summary_agent
from agents.market_agent import get_market_agent
from agents.risk_agent import get_risk_agent
from agents.decision_agent import get_decision_agent
from agents.report_agent import get_report_agent

# 환경 설정
load_dotenv()
st.set_page_config(page_title="📊 스타트업 투자 분석", layout="wide")
st.title("📈 AI 기반 스타트업 투자 평가")

# 입력: 검색어
query = st.text_input("분석할 기업명 또는 키워드 입력", value="보이저엑스")

if st.button("투자 분석 시작"):
    with st.spinner("Pinecone에서 문서 검색 중..."):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "investment-analysis-final"
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

    with st.spinner("에이전트 실행 중..."):
        summary_chain = get_summary_agent()
        market_chain = get_market_agent()
        risk_chain = get_risk_agent()
        decision_chain = get_decision_agent()
        report_chain = get_report_agent()

        class AgentState(TypedDict, total=False):
            context: str
            summary_result: str
            market_result: str
            risk_result: str
            decision_result: str
            report_result: str

        builder = StateGraph(AgentState)
        builder.add_node("summary", lambda state: {
            "summary_result": summary_chain.run({"context": state["context"]})
        })
        builder.add_node("market_and_risk", lambda state: {
            "market_result": market_chain.run({
                "context": state["context"],
                "summary": state["summary_result"]
            }),
            "risk_result": risk_chain.run({"context": state["context"]})
        })
        builder.add_node("decision", lambda state: {
            "decision_result": decision_chain.run({
                "summary": state["summary_result"],
                "market": state["market_result"],
                "risk": state["risk_result"]
            })
        })
        builder.add_node("report", lambda state: {
            "report_result": report_chain.run({
                "summary": state["summary_result"],
                "market": state["market_result"],
                "risk": state["risk_result"],
                "decision": state["decision_result"]
            })
        })

        builder.set_entry_point("summary")
        builder.add_edge("summary", "market_and_risk")
        builder.add_edge("market_and_risk", "decision")
        builder.add_edge("decision", "report")
        builder.add_edge("report", END)

        graph = builder.compile()
        result = graph.invoke({"context": context})

    # 결과 출력
    st.subheader("📝 기업 요약")
    st.markdown(result["summary_result"])

    st.subheader("📊 시장성 분석")
    st.markdown(result["market_result"])

    st.subheader("⚠️ 리스크 분석")
    st.markdown(result["risk_result"])

    st.subheader("📌 투자 판단")
    st.markdown(result["decision_result"])

    st.subheader("🗂️ 보고서")
    st.markdown(result["report_result"])

    # PDF 저장 및 다운로드
    os.makedirs("outputs", exist_ok=True)
    filename = os.path.join("outputs", f"{query}_투자보고서.pdf")
    save_report_to_pdf(result["report_result"], filename)
    with open(filename, "rb") as f:
        st.download_button("📄 PDF 다운로드", data=f, file_name=os.path.basename(filename), mime="application/pdf")
