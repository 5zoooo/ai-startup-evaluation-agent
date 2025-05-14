from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from prompts.risk_prompt import RISK_PROMPT

def get_risk_agent():
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=RISK_PROMPT, llm=llm)

