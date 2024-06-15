from typing import List

from langchain.schema import Document
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_arxiv_docs(
    query: str, num_docs: int = 5, doc_content_chars_max: int = 5000
) -> List[Document]:
    loader = ArxivLoader(
        query,
        load_max_docs=num_docs,
        doc_content_chars_max=doc_content_chars_max,
        load_all_available_meta=False,
    )

    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )

    documents = text_splitter.split_documents(raw_documents)

    return documents
