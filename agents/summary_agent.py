# summary_agent.py
import os
from collections import defaultdict
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain.tools import Tool

# 1. 환경 변수 로딩
load_dotenv()

# 2. Pinecone 연결
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "investment-analysis-final"

# 3. 임베딩 모델
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 4. 벡터 DB 연결
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

class SummaryAgent:
    def __init__(self, vectorstore, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm or ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

    def search_and_group(self, query="보이저엑스 기업 정보 전체 요약", k=20):
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)
        grouped_docs = defaultdict(list)
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "unknown_source")
            grouped_docs[source].append(doc.page_content)
        grouped_texts = {source: "\n\n".join(contents) for source, contents in grouped_docs.items()}
        return grouped_texts

    def build_prompt(self, grouped_texts):
        prompt = f"""
다음 4가지 PDF 기반 정보를 바탕으로 투자자 관점에서 기업 요약을 해주세요.

[기업 소개서]
{grouped_texts.get('voyagerx_intro', '없음')}

[뉴스 기사]
{grouped_texts.get('ai_news', '없음')}

[유튜브 STT]
{grouped_texts.get('video_stt_script', '없음')}

[스타트업 평가 지표]
{grouped_texts.get('startup_metrics', '없음')}

주의: 자연스러운 한국어로 핵심을 요약하세요. 분량은 200자 이내로 요약해주세요.
"""
        return prompt

    def run(self, query="보이저엑스 기업 정보 전체 요약"):
        grouped_texts = self.search_and_group(query)
        prompt_text = self.build_prompt(grouped_texts)
        response = self.llm.invoke(prompt_text)
        return response.content  # 자연어 문자열 반환

def create_summary_tool(vectorstore):
    agent = SummaryAgent(vectorstore)

    def summary_tool_fn(query: str) -> str:
        result = agent.run(query)
        return result  # JSON 없이 자연어 문자열 반환

    summary_tool = Tool.from_function(
        func=summary_tool_fn,
        name="SummaryAgentTool",
        description="기업 PDF 벡터 DB에서 검색 후 투자자 관점 기업 요약을 자유 텍스트로 반환하는 Agent. 입력은 쿼리 문자열."
    )

    return summary_tool

# Tool 예시 호출
tool = create_summary_tool(vectorstore)
result = tool.invoke("보이저엑스 기업 정보 전체 요약")
print("Summary Agent Tool 결과:\n", result)