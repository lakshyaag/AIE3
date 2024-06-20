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

    async for event in graph.astream_events(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        version="v2",
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"]["langgraph_node"] == "generate"
        ):
            await msg.stream_token(event["data"]["chunk"].content)

    await msg.send()
