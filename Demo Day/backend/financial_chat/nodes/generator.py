from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from financial_chat.prompts import GENERATOR_PROMPT


prompt = ChatPromptTemplate.from_messages([("human", GENERATOR_PROMPT)])
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, streaming=True)

generator_chain = prompt | llm
