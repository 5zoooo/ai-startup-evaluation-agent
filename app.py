from typing import TypedDict
import os
import markdown2  # 마크다운 → HTML 변환
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import StateGraph, END
from weasyprint import HTML  # ✅ HTML to PDF 변환용

# 에이전트 임포트
from agents.summary_agent import get_summary_agent
from agents.market_agent import get_market_agent
from agents.risk_agent import get_risk_agent
from agents.decision_agent import get_decision_agent
from agents.report_agent import get_report_agent

# ✅ 보고서 저장 함수 (WeasyPrint 기반)
def save_report_to_pdf(markdown_text: str, filename: str = "startup_report.pdf"):
    html_content = markdown2.markdown(markdown_text)

    styled_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Nanum Gothic', sans-serif;
                line-height: 1.6;
                margin: 2em;
            }}
            h1 {{ font-size: 22pt; font-weight: bold; margin-top: 30px; }}
            h2 {{ font-size: 18pt; font-weight: bold; margin-top: 20px; }}
            ul {{ margin-left: 1.5em; }}
            li {{ margin-bottom: 6px; }}
            p {{ margin: 6px 0; font-size: 12pt; }}
            strong {{ font-weight: bold; }}
        </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    HTML(string=styled_html).write_pdf(filename)
    print(f"📄 PDF 저장 완료: {filename}")

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
report_chain = get_report_agent()

# ✅ LangGraph 정의
class AgentState(TypedDict, total=False):
    context: str
    summary_result: str
    market_result: str
    risk_result: str
    decision_result: str
    report_result: str

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

# 4. 보고서 생성
builder.add_node("report", lambda state: {
    "report_result": report_chain.run({
        "summary": state["summary_result"],
        "market": state["market_result"],
        "risk": state["risk_result"],
        "decision": state["decision_result"]
    })
})

# ✅ 그래프 흐름 정의
builder.set_entry_point("summary")
builder.add_edge("summary", "market_and_risk")
builder.add_edge("market_and_risk", "decision")
builder.add_edge("decision", "report")
builder.add_edge("report", END)

# ✅ 실행
graph = builder.compile()
result = graph.invoke({"context": context})

# ✅ 출력
print("📝 요약:\n", result["summary_result"])
print("\n📊 시장성:\n", result["market_result"])
print("\n⚠️ 리스크:\n", result["risk_result"])
print("\n📌 판단:\n", result["decision_result"])
print("\n🗂️ 보고서:\n", result["report_result"])


# ✅ PDF 저장 (outputs 디렉터리로)
output_path = os.path.join("outputs", "보이저엑스_투자보고서.pdf")
os.makedirs("outputs", exist_ok=True)
save_report_to_pdf(result["report_result"], output_path)