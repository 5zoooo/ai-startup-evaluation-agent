# agent_2_market.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def get_market_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 스타트업 시장 전문가야. 기술을 바탕으로 시장성을 분석해줘."),
        ("human", "다음 문서를 참고해서 시장 가능성과 성장성을 평가해줘:\n\n{context}")
    ])
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=prompt, llm=llm)