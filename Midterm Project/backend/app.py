import os
import chainlit as cl
from dotenv import load_dotenv
from graph import create_graph
from langchain_core.runnables import RunnableConfig
from starters import set_starters

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
graph = create_graph()


@cl.on_message
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.
    """

    msg = cl.Message(content="")

    res = graph.invoke(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )

    print(res)

    msg.content = res["generation"].content

    await msg.send()
