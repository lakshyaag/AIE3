from langchain_core.pydantic_v1 import BaseModel


class ChatInputType(BaseModel):
    question: str
