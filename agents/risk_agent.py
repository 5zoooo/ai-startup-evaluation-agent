# agent_3_risk.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def get_risk_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 투자 심사역이야. 기업의 리스크를 분석해줘."),
        ("human", "다음 문서를 바탕으로 다음과 같은 항목별로 리스크를 평가해줘:\n\n1. 기술 리스크\n2. 시장 리스크\n3. 조직 리스크\n4. 기타 우려 사항\n\n문서:\n{context}")
    ])
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=prompt, llm=llm)
