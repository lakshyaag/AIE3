from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
import os

QDRANT_DATA_DIR = "./qdrant_data"

if os.path.exists(QDRANT_DATA_DIR):
    
    vectorstore = Qdrant.from_existing_collection(
        embedding=OpenAIEmbeddings(),
        path=QDRANT_DATA_DIR,
        collection_name="rag-chroma",
    )

else:
    files = ["./data/airbnb_10q_q1.pdf"]
    # Load documents
    docs = [PyPDFLoader(file).load() for file in files]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    print(f"Number of document splits: {len(doc_splits)}")

    # Add to vectorDB with on-disk storage
    vectorstore = Qdrant.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        path="./qdrant_data",  # Local mode with on-disk storage
        embedding=OpenAIEmbeddings(),
    )

retriever = vectorstore.as_retriever()

# print(retriever.invoke("What is the average length of stay for Airbnb guests?"))