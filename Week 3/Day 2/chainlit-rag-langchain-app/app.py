import os
from operator import itemgetter

import chainlit as cl
from dotenv import find_dotenv, load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rich import print
from langchain_community.vectorstores import Qdrant
import loader

load_dotenv(find_dotenv())

RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant that can answer questions about ArXiv papers in the domain of {question}. 
You can provide detailed answers to questions about the content of the papers, the authors, the publication date, and more.
If the relevant information is not present in the papers, say "I don't know". 
"""


@cl.on_chat_start
async def on_chat_start():
    assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY in .env file"

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    query = await cl.AskUserMessage(
        content="What ArXiv papers would you like to talk about?"
    ).send()

    if query:
        docs = loader.get_arxiv_docs(
            query["content"], num_docs=5, doc_content_chars_max=5000
        )

        if not docs:
            await cl.ErrorMessage(content="No documents found").send()
            return

        vectorstore = Qdrant.from_documents(
            docs,
            embedding=embedding_model,
            location=":memory:",
            collection_name="ArXiV papers",
        )

        retriever = vectorstore.as_retriever()

        await cl.Message(content="Vector index created!").send()

        cl.user_session.set("retriever", retriever)

        model = ChatOpenAI(model="gpt-3.5-turbo")
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

        retrieval_augmented_qa_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | model, "context": itemgetter("context")}
        )

        cl.user_session.set("embedding_model", embedding_model)
        cl.user_session.set("runnable", retrieval_augmented_qa_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    if cl.user_session.get("retriever"):
        msg = cl.Message(content="")
        context = None

        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            if chunk.get("context", None):
                context = chunk.get("context")

            else:
                await msg.stream_token(chunk.get("response").content)

        await cl.Message(content=print_context(context)).send()

        await msg.send()

    else:
        await cl.ErrorMessage(
            content="No documents found in index. Please restart the session."
        ).send()

        return


def print_context(context):
    msg = ""
    for d in context:
        msg += f"Content: {d.page_content[:100]}...{d.page_content[-100:]}\n"

        msg += "\n\n"

    return msg
