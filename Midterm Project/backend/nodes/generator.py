from utils.prompts import GENERATOR_PROMPT
from utils.config import GENERATOR_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Prompt
prompt = ChatPromptTemplate.from_messages([("human", GENERATOR_PROMPT)])

prompt

# Generator LLM
llm = ChatOpenAI(model_name=GENERATOR_MODEL, temperature=0.7, streaming=True)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm

# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)
