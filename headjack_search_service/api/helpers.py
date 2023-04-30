"""
Module containing all config related things
"""
import logging

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

_logger = logging.getLogger(__name__)

def get_chroma_client():  # pragma: no cover
    """
    Get a chromadb client
    """
    chroma_client = chromadb.Client(
        Settings(
            chroma_api_impl="rest",
            chroma_server_host="hss-chromadb",
            chroma_server_http_port="16411",
        ),
    )
    return chroma_client

def get_collection(client: chromadb.Client, collection: str) -> Collection:
    """
    Get the headjack chroma collection
    """
    _logger.info(f"Getting chroma collection {collection}")
    return client.get_or_create_collection(
        collection,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        ),
    )