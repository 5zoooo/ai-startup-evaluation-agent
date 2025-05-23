from langchain.prompts import ChatPromptTemplate

MARKET_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 벤처캐피탈 투자심사자야. 스타트업의 기술과 산업 배경을 바탕으로 시장성을 평가해야 해."),
    ("human", """아래는 스타트업의 핵심 기술과 제품 요약, 그리고 산업 정보야.  
이걸 참고해서 이 스타트업이 진출하려는 시장의 시장성이 얼마나 있는지 평가해줘.  
아래 항목별로 하나하나 3줄 이상씩 자세히 설명해줘.

### [시장성 평가 항목]

1. **시장 규모**  
   - 이 기술이 주로 사용될 산업 분야는 어디야?  
   - 그 산업의 전체 시장 규모는 지금 어느 정도 되는지도 알려줘.

2. **성장 가능성**  
   - 해당 시장은 얼마나 빨리 성장 중이야 (예: CAGR)?  
   - 최근 2~3년 사이에 어떤 변화나 기회가 있었는지도 알려줘.

3. **경쟁 강도**  
   - 이 시장에 비슷한 기술이나 제품을 가진 경쟁사가 있다면 누구야?  
   - 이 스타트업이 어떤 점에서 차별화되는지도 설명해줘.

4. **진입 장벽**  
   - 이 시장에 새로 들어오려면 어떤 자원(기술, 인력, 인증 등)이 필요해?  
   - 규제나 인증처럼 진입을 어렵게 만드는 게 있는지도 같이 말해줘.

5. **산업 트렌드 적합성**  
   - 이 기술이 최근 산업 흐름이나 정부 정책, 글로벌 트렌드에 얼마나 잘 맞아?  
   - 예를 들면 ESG, 생성형 AI, 자동화, 원격화 같은 키워드랑 연결해서 얘기해줘.

6. **종합 평가**  
   - 지금까지 항목들을 종합했을 때, 이 스타트업은 시장 측면에서 투자할 만한 가치가 있어 보여?  
   - 그렇게 생각하는 이유도 같이 말해줘.

[기업 요약]
{summary}

[산업 정보]
{context}
""")
])