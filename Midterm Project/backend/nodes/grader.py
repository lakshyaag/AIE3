from utils.prompts import GRADER_SYSTEM_PROMPT, GRADER_PROMPT
from utils.config import GRADER_MODEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


## Grading model
class DocumentGrade(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description='Document is relevant to the question, "yes" or "no"'
    )
    reasoning: str = Field(description="Reasoning for the score")


# Grader Prompts
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADER_SYSTEM_PROMPT),
        ("human", GRADER_PROMPT),
    ]
)


# LLM with function call
llm = ChatOpenAI(model=GRADER_MODEL, temperature=0, streaming=True)

structured_llm_grader = llm.with_structured_output(DocumentGrade)


retrieval_grader = grade_prompt | structured_llm_grader

# question = "agent memory"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
