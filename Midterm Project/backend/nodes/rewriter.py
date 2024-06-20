### Rewrite a question to be more optimized for Wikipedia search
from utils.config import REWRITER_MODEL
from utils.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_SYSTEM_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Question rewriter LLM
llm = ChatOpenAI(model=REWRITER_MODEL, temperature=0, streaming=True)

# Prompt
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_REWRITE_SYSTEM_PROMPT),
        ("human", QUERY_REWRITE_PROMPT),
    ]
)

question_rewriter = rewrite_prompt | llm | StrOutputParser()
# question_rewriter.invoke({"question": question})
