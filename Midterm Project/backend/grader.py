import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


## Grading model
class DocumentGrade(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description='Document is relevant to the question, "yes" or "no"'
    )


# Grader Prompts
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)


# LLM with function call
llm = ChatOpenAI(model=config.GRADER_MODEL, temperature=0, streaming=True)

structured_llm_grader = llm.with_structured_output(DocumentGrade)
structured_llm_grader


retrieval_grader = grade_prompt | structured_llm_grader

# question = "agent memory"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
