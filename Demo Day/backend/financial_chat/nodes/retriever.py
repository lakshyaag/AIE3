from financial_chat.supabase_client import vectorstore
from langchain.chains.query_constructor.base import (
    AttributeInfo,
)
from langchain_community.query_constructors.supabase import SupabaseVectorTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="ticker",
        description="The ticker of the company",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year of the report",
        type="integer or list[integer]",
    ),
]
document_content_description = "Chunks of text from financial reports"

llm = ChatOpenAI(temperature=0, model="gpt-4o")

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    structured_query_translator=SupabaseVectorTranslator(),
)
