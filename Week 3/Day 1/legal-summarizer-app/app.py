import os

import chainlit as cl  # importing chainlit for our app
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

import bitsandbytes as bnb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Prompt Templates
INSTRUCTION_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Please convert the following legal content into a human-readable summary<|eot_id|><|start_header_id|>user<|end_header_id|>
[LEGAL_DOC]
{input}
[END_LEGAL_DOC]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

RESPONSE_TEMPLATE = """
{summary}<|eot_id|>
"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

config = PeftConfig.from_pretrained("lakshyaag/llama38binstruct_summarize")

base_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "lakshyaag/llama38binstruct_summarize")

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("lakshyaag/llama38binstruct_summarize")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def create_prompt(sample, include_response=False):
    """
    Parameters:
      - sample: dict representing row of dataset
      - include_response: bool
    Functionality:
      This function should build the Python str `full_prompt`.
      If `include_response` is true, it should include the summary -
      else it should not contain the summary (useful for prompting) and testing
    Returns:
      - full_prompt: str
    """

    full_prompt = INSTRUCTION_PROMPT_TEMPLATE.format(input=sample["original_text"])

    if include_response:
        full_prompt += RESPONSE_TEMPLATE.format(summary=sample["reference_summary"])

    full_prompt += "<|end_of_text|>"

    return full_prompt


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    prompt = create_prompt({"original_text": message.content}, include_response=False)

    # convert str input into tokenized input
    encoded_input = tokenizer(prompt, return_tensors="pt")

    # send the tokenized inputs to our GPU
    model_inputs = encoded_input.to("cuda" if torch.cuda.is_available() else "cpu")

    # generate response and set desired generation parameters
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # decode output from tokenized output to str output
    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # return only the generated response (not the prompt) as output
    response = decoded_output[0].split("<|end_header_id|>")[-1]

    await message.reply(response)


if __name__ == "__main__":
    cl.run()
