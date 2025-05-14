# agent_2_market.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from prompts.market_prompt import MARKET_PROMPT

def get_market_agent():
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=MARKET_PROMPT, llm=llm)