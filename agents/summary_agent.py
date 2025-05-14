# summary_agent.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def get_summary_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 기업 분석 전문가야. 기업의 핵심 기술과 개요를 요약해줘."),
        ("human", "다음 문서를 보고 핵심 요약을 작성해줘:\n\n{context}")
    ])
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=prompt, llm=llm)