# agent_5_report.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def get_report_agent():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            너는 벤처캐피탈 심사역에게 제출될 투자 분석 보고서를 작성하는 전문 분석가야.
            아래 항목을 참고해서 객관적이고 논리적인 보고서를 작성해.
            보고서는 명확한 제목과 항목별 소제목으로 구성하고, 각 내용을 3~5문장으로 구체적으로 설명해줘.
            문체는 간결하고 비즈니스적이며 판단 근거가 잘 드러나도록 해.
            """
        ),
        (
            "human",
            """
            다음 내용을 기반으로 투자 보고서를 작성해줘:

            [기업 요약]
            {summary}

            [시장성 평가]
            {market}

            [리스크 분석]
            {risk}

            [최종 판단]
            {decision}
            """
        )
    ])
    llm = ChatOpenAI(temperature=0.3)
    return LLMChain(prompt=prompt, llm=llm)

