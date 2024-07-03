import pytest
from financial_chat.nodes.retriever import retriever
from financial_chat.nodes.generator import generator_chain


def test_generator():
    query = "What was NVDA's total revenue in 2022?"
    docs = retriever.invoke(query)

    response = generator_chain.invoke({"question": query, "context": docs})

    assert "26.91 billion" in response.content


def test_generator_2():
    query = "What is COST's business model?"
    docs = retriever.invoke(query)

    response = generator_chain.invoke({"question": query, "context": docs})

    assert "membership" in response.content
