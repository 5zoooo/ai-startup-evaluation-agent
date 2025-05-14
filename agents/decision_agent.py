# agent_4_decision.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def get_decision_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 벤처캐피탈 심사위원이야. 종합적으로 투자 결정을 내려야 해."),
        ("human", "다음 내용을 참고해서 이 기업에 투자할지 여부를 판단해줘. 판단 이유도 함께 작성해줘:\n\n요약: {summary}\n시장성: {market}\n리스크: {risk}")
    ])
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=prompt, llm=llm)