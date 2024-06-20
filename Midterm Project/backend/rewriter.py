### Rewrite a question to be more optimized for Wikipedia search
import config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Question rewriter LLM
llm = ChatOpenAI(model=config.REWRITER_MODEL, temperature=0, streaming=True)

# Prompt
system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for searching on Wikipedia. Look at the input and try to reason about the underlying semantic intent / meaning, such that the new question is more optimized for search."""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = rewrite_prompt | llm | StrOutputParser()
# question_rewriter.invoke({"question": question})
