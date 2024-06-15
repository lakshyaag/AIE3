import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from dotenv import load_dotenv

load_dotenv()

model_name = "lakshyaag/llama38binstruct_summarize"
config = PeftConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3-8B-Instruct"
)
model = PeftModel.from_pretrained(base_model, model_name)

# Prompt Templates
system_template = """Your task is to provide a summary of the provided document."""

user_template = """{input}"""


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    inputs = tokenizer.encode(
        message,
        return_tensors="pt",
    )

    outputs = model.generate(
        inputs,
        max_length=100,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


if __name__ == "__main__":
    cl.run()
