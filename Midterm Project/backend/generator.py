### Generate
import config
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Prompt
agent_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"""

prompt = ChatPromptTemplate.from_messages([("human", agent_prompt)])

prompt

# Generator LLM
llm = ChatOpenAI(model_name=config.GENERATOR_MODEL, temperature=0, streaming=True)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)
