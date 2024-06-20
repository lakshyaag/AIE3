from typing import List

from langchain.schema import Document
from langgraph.graph import END, StateGraph
from nodes.generator import rag_chain
from nodes.grader import retrieval_grader
from nodes.retriever import retriever
from nodes.rewriter import question_rewriter
from tools.search_wikipedia import wikipedia
from typing_extensions import TypedDict


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
    generation: str
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


# DEFINE CONDITIONAL EDGES
def generate_or_not(state):
    print("Determining whether to query Wikipedia...")

    wiki_search = state["wiki_search"]
    filtered_docs = state["documents"]

    if len(filtered_docs) == 0 and wiki_search:
        print("Rewriting query and supplementing information from Wikipedia...")
        return "rewrite_query"

    else:
        print("Relevant documents found.")
        return "generate"


def create_graph():
    # DEFINE WORKFLOW
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("search_wikipedia", search_wikipedia)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        generate_or_not,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )

    workflow.add_edge("rewrite_query", "search_wikipedia")
    workflow.add_edge("search_wikipedia", "generate")
    workflow.add_edge("generate", END)

    # COMPILE GRAPH
    app = workflow.compile()

    return app
