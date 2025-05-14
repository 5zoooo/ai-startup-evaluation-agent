# summary_agent.py
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from prompts.summary_prompt import SUMMARY_PROMPT

def get_summary_agent():
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=SUMMARY_PROMPT, llm=llm)
