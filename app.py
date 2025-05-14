# app.py
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
import os
from summary_agent import SummaryAgent

# 1. Pinecone 연결
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "investment-analysis-final"

# 2. 임베딩 모델
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 3. 벡터 DB 연결
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

# Tool 등록 함수
def create_summary_tool(vectorstore):
    agent = SummaryAgent(vectorstore)

    def summary_tool_fn(query: str) -> str:
        result = agent.run(query)
        return json.dumps(result, ensure_ascii=False, indent=2)

    summary_tool = Tool.from_function(
        func=summary_tool_fn,
        name="SummaryAgentTool",
        description="기업 PDF 벡터 DB에서 검색 후 투자자 관점 기업 요약을 반환하는 Agent. 입력은 쿼리 문자열."
    )

    return summary_tool

# Tool 예시 호출
tool = create_summary_tool(vectorstore)
result = tool.invoke("보이저엑스 기업 정보 전체 요약")
print("✅ Summary Agent Tool 결과:\n", result)
