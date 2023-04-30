"""
Headjack search service
"""
from enum import Enum
import logging
from typing import List

from chromadb.api.local import LocalAPI
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from pydantic import BaseModel
from headjack_search_service.api.helpers import get_chroma_client
from fastapi.templating import Jinja2Templates
import asyncio
from chromadb.utils import embedding_functions

_logger = logging.getLogger(__name__)

class COLLECTION_TYPE(str, Enum):
    knowledge = "knowledge"
    metrics = "metrics"
    
app = FastAPI()

@app.get("/healthcheck/")
async def health_check(*, chroma_client: LocalAPI = Depends(get_chroma_client)):
    chroma_client.heartbeat()
    return {"status": "OK"}

@app.get("/query/")
async def query(text: str, collection: COLLECTION_TYPE, n: int = 3, *, chroma_client: LocalAPI = Depends(get_chroma_client)):
    _logger.info(f"Pulling collection for {collection.value}")
    chroma_collection = chroma_client.get_collection(collection.value, embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        ),)
    results = chroma_collection.query(query_texts=[text], n_results=n)
    return results  
