from typing import List

import aiosqlite
from generator import rag_chain
from grader import retrieval_grader
from langchain.schema import Document
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from retriever import retriever
from rewriter import question_rewriter
from tools.search_wikipedia import wikipedia
from typing_extensions import Annotated, TypedDict


# DEFINE STATE GRAPH
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    # Update this to work with memory in a better way.
    question: str
    generation: Annotated[List[str], add_messages]
    wiki_search: str
    documents: List[str]


# DEFINE NODES
def retrieve(state):
    print("Retrieving documents...")

    question = state["question"]

    docs = retriever.invoke(question)

    return {"question": question, "documents": docs}


def generate(state):
    print("Generating answer...")

    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    print("Grading documents...")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    search_wikipedia = False

    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )

        grade = score.binary_score

        if grade == "yes":
            print("Document is relevant to the question.")
            filtered_docs.append(doc)
        else:
            print("Document is not relevant to the question.")
            search_wikipedia = True

            continue

    return {
        "documents": filtered_docs,
        "question": question,
        "wiki_search": search_wikipedia,
    }


def rewrite_query(state):
    print("Rewriting question...")

    question = state["question"]
    documents = state["documents"]

    rewritten_question = question_rewriter.invoke({"question": question})

    return {"question": rewritten_question, "documents": documents}


def search_wikipedia(state):
    print("Searching Wikipedia...")

    question = state["question"]
    documents = state["documents"]

    wiki_search = wikipedia.invoke(question)

    wiki_results = Document(page_content=wiki_search)

    documents.append(wiki_results)

    return {"question": question, "documents": documents}


def question_node(state):
    question = state["question"]
    generation = state.get("generation", None)

    return {"question": question, "generation": generation}


def check_memory_or_db(state):
    print("Checking memory or database...")
    print(state)

    if state.get("generation", None) is not None and len(state["generation"]) > 0:
        question = state["question"]
        generation = state["generation"]

        perform_rag = retrieval_grader.invoke(
            {"question": question, "document": generation}
        )

        if perform_rag.binary_score == "yes":
            print("Memory not sufficient. Performing RAG.")
            return "retrieve"

        else:
            print("Memory sufficient. Generating answer.")
            return "generate"

    else:
        print("No memory found. Retrieving documents...")
        return "retrieve"


# DEFINE CONDITIONAL EDGES
def generate_or_not(state):
    print("Determining whether to query Wikipedia...")

    wiki_search = state["wiki_search"]

    if wiki_search:
        print("Rewriting query and supplementing information from Wikipedia...")
        return "rewrite_query"

    else:
        print("Relevant documents found.")
        return "generate"


def create_graph():
    # DEFINE WORKFLOW
    workflow = StateGraph(GraphState)

    workflow.add_node("start", question_node)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("search_wikipedia", search_wikipedia)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("start")
    workflow.add_conditional_edges(
        "start", check_memory_or_db, {"retrieve": "retrieve", "generate": "generate"}
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        generate_or_not,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )

    workflow.add_edge("rewrite_query", "search_wikipedia")
    workflow.add_edge("search_wikipedia", "generate")
    workflow.add_edge("generate", END)

    # DEFINE MEMORY
    checkpoints = aiosqlite.connect("./checkpoints/checkpoint.sqlite")
    memory = AsyncSqliteSaver(checkpoints)

    # COMPILE GRAPH
    app = workflow.compile(checkpointer=memory)

    return app
