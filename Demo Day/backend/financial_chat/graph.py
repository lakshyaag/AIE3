import logging
from typing import List, Annotated, Optional, Literal
from typing_extensions import TypedDict

from operator import add
from langgraph.graph import StateGraph, add_messages, END
from langchain_core.messages import AnyMessage
from financial_chat.nodes import report_params_generator
from financial_chat.nodes import retriever
from financial_chat.nodes import generator
from financial_chat.chunk_load import (
    check_if_documents_exist,
    chunk_items,
    load_documents,
)
from financial_chat.fetch_documents import get_company_filing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# STATE
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    documents: Annotated[Optional[List[str]], add]
    report_params: Optional[report_params_generator.ReportParams]


# NODES
def generate_params(state: State) -> State:
    logger.info("Generating parameters...")

    query = state["messages"][-1].content

    report_params = report_params_generator.chain.invoke({"input": query})

    if check_if_documents_exist(report_params):
        logger.info("Documents exist in DB. Skipping fetch...")
        return {}

    else:
        logger.info("Documents do not exist in DB. Fetching...")
        chunks = chunk_items(get_company_filing(report_params)["items"], report_params)
        load_documents(chunks)

    return {"report_params": report_params}


def retrieve(state: State) -> State:
    logger.info("Retrieving documents...")

    query = state["messages"][-1].content

    docs = retriever.retriever.invoke(query)
    logger.info(f"Documents retrieved: {docs}")

    return {"documents": docs}


def generate(state: State) -> State:
    logger.info("Generating answer...")

    question = state["messages"][-1].content
    documents = state["documents"]

    generation = generator.generator_chain.invoke(
        {"context": documents, "question": question}
    )

    return {"messages": generation}


# CONDITIONAL EDGE
def fetch_docs(state: State) -> Literal["fetch_documents", "generate"]:
    logger.info("Checking if documents need to be fetched...")
    logger.info(f"State: {state}")

    if len(state["documents"]) == 0 or state["documents"][-1] is None:
        logger.info("Relevant documents not found. Fetching...")
        return "fetch_documents"
    else:
        return "generate"


def create_graph():
    workflow = StateGraph(State)

    workflow.add_node("generate_params", generate_params)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        fetch_docs,
        {
            "fetch_documents": "generate_params",
            "generate": "generate",
        },
    )

    workflow.add_edge("generate_params", "retrieve")
    workflow.add_edge("generate", END)

    graph = workflow.compile()

    return graph
