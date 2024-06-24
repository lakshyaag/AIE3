import os

import streamlit as st
from dotenv import load_dotenv
from graph import create_graph

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
graph = create_graph()

st.set_page_config(page_title="Airbnb 10-Q Chatbot", layout="wide", page_icon="🖋️")

st.title("Airbnb 10-Q Chatbot")

st.markdown(
    """
    ### This chatbot is designed to answer questions about Airbnb's 10-Q financial statement for the quarter ended March 31, 2024.
    
    It uses corrective RAG as the architecture to retrieve information based on the user's input, grading the relevance of the retrieved documents, and optionally querying Wikipedia for additional information, if necessary.
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "Assistant",
            "content": "Hello! I'm here to help you with Airbnb's 10-Q. Feel free to ask me anything.",
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


if prompt := st.chat_input("Ask about Airbnb's 10-Q"):
    st.chat_message("User").write(prompt)

    st.session_state.messages.append({"role": "User", "content": prompt})

    with st.chat_message("Assistant"):
        with st.status("Thinking...", expanded=True) as status:
            for event in graph.stream(
                {
                    "question": prompt,
                },
            ):
                for key, value in event.items():
                    if key == "generate":
                        st.write(value["generation"].content)

            status.update(label="Done!", expanded=True, state="complete")

    st.session_state.messages.append(
        {"role": "Assistant", "content": event["generate"]["generation"].content}
    )
