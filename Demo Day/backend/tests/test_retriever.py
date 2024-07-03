import pytest
from financial_chat.nodes.retriever import retriever


def test_retriever():
    response = retriever.invoke("What was NVDA's revenue in 2022?")

    print(response)

def test_retriever_2():
    response = retriever.invoke("What are the key risks to COST?")

    print(response)
