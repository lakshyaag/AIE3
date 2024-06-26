from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from graph import create_graph
from utils.chat_types import ChatInputType

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="CRAG Backend",
    version="1.0",
    description="Backend to run agent performing corrective RAG over annual reports",
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = create_graph()


runnable = graph.with_types(input_type=ChatInputType, output_type=dict)

add_routes(app, runnable, path="/chat", playground_type="default")
