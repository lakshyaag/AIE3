from textwrap import dedent

GENERATOR_PROMPT = dedent(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Question: {question}

Context: {context}

Answer:"""
)
