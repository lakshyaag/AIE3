from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

from financial_chat.supabase_client import vectorstore, supabase
from financial_chat.types_ import ReportParams


def chunk_items(
    items: dict[str, str],
    params: ReportParams,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Chunk items into documents for processing.

    Args:
        items (dict[str, str]): The items to chunk.
        ticker (str): The ticker of the company.
        year (int): The year of the filing.
        chunk_size (int, optional): The size of the chunks. Defaults to 1024.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 100.

    Returns:
        List[Document]: The documents.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    documents = []

    for item_name, item in items.items():
        chunks = text_splitter.split_text(item)

        chunks = [
            Document(
                page_content=chunk,
                metadata={
                    "item_name": item_name,
                    "ticker": params.ticker,
                    "form": params.form,
                    "year": params.year,
                },
            )
            for chunk in chunks
        ]

        documents.extend(chunks)

    return documents


def check_if_documents_exist(params: ReportParams) -> bool:
    """
    Check if documents already exist in the vector store.

    Args:
        documents (List[Document]): The documents to check.

    Returns:
        bool: True if the documents exist, False otherwise.
    """
    rows = (
        supabase.table("documents_duplicate")
        .select("*", count="exact")
        .eq("metadata->>ticker", params.ticker)
        .eq("metadata->year", params.year)
        .eq("metadata->>form", "10-K")
        .in_("metadata->>item_name", [item.value for item in params.item_names])
        .execute()
    )

    return rows.count > 0


def load_documents(documents: List[Document]) -> None:
    """
    Load documents into the vector store.

    Args:
        documents (List[Document]): The documents to load.
    """

    try:
        vectorstore.add_documents(documents)
    except Exception as e:
        raise RuntimeError("Failed to load documents into the vector store") from e
