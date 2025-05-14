# summary_agent.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def get_summary_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 기업 분석 전문가이며 투자자 관점에서 기업의 주요 정보를 정확하고 간결하게 요약해야 해. 기업의 핵심 기술, 주요 제품, 비즈니스 모델, 성장 가능성, 경쟁력 등의 항목을 중심으로 투자자가 빠르게 이해할 수 있도록 핵심만 정리해."),
        ("human", 
        """다음 문서를 분석하여 아래 기준에 따라 한국어로 간결하게 요약해줘.

[요약 기준]
- 기업명 및 설립 연도 (있는 경우)
- 핵심 제품/서비스 및 주요 기능
- 핵심 기술과 기술적 차별성
- 비즈니스 모델 및 수익 구조 (있는 경우)
- 시장 타겟 및 성장 가능성
- 경쟁사 대비 차별점

[분석 문서]
{context}

주의:
- 투자자가 빠르게 이해할 수 있도록 핵심만 요약.
- 문장 중심으로 자연스럽게 요약하되, 정보가 빠지지 않게 주의.
- 표나 리스트 없이 서술형으로 작성.
""")
    ])
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=prompt, llm=llm)
