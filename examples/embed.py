"""
Script to load knowledge documents into a vector database
"""
import argparse
import logging
import re
import sys
from glob import glob
from pathlib import Path

import chromadb
import requests
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s  ", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)
_logger = logging.getLogger("embed.py")


def get_chroma_client(host: str, port: str) -> chromadb.Client:
    """
    Get a chroma client
    """
    _logger.info(f"Getting chroma client for host {host} and port {port}")
    return chromadb.Client(
        Settings(
            chroma_api_impl="rest",
            chroma_server_host=host,
            chroma_server_http_port=port,
        ),
    )


def window_document(file_name: str, document_text: str, window_size: int = 200, overlap: int = 50):
    """
    Splits a document into overlapping windows of fixed size.

    Args:
        document (str): The document to split.
        window_size (int): The word size of each window.
        overlap (int): The amount of word overlap between adjacent windows.

    Returns:
        List[str]: A list of overlapping windows.
    """
    document = re.split(r"\s+", document_text)
    title = re.split(r"[._-]+", file_name) + re.split(r"\s+", document_text.split("\n")[0])[:10]
    windows = []
    start = 0
    end = window_size
    while end <= len(document):
        windows.append(" ".join((title if start != 0 else []) + document[start:end]))
        start += window_size - overlap
        end += window_size - overlap
    if end > len(document) and start < len(document):
        windows.append(" ".join(title + document[start:]))
    return windows


def index_metrics(dj_url: str, client: chromadb.Client):
    """
    Request metrics from a DataJunction server and embed the descriptions into chroma
    """
    _logger.info(f"Indexing DataJunction metrics ({dj_url})")
    metrics_collection = client.get_or_create_collection(
        "metrics",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"),
    )

    # Get DJ metrics
    metrics_list = requests.get(
        f"{dj_url}/metrics",
    ).json()
    documents = []
    metadatas = []
    ids = []
    for metric in metrics_list:
        _logger.info(f"Indexing metric {metric}")
        m = requests.get(
            f"{dj_url}/metrics/{metric}/",
        ).json()
        documents.append(m["description"])
        metadatas.append(
            {
                "name": m["name"],
                "query": m["query"],
            }
        )
        ids.append(str(m["id"]))

    metrics_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


def index_knowledge(knowledge_dir: str, client: chromadb.Client):
    """
    Read text documents in a directory and embed the content into chroma
    """
    _logger.info(f"Indexing knowledge documents ({knowledge_dir})")
    knowledge_collection = client.get_or_create_collection(
        "knowledge",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"),
    )
    knowledge_files = glob(knowledge_dir)
    knowledge_doc_texts = {}
    for kd in knowledge_files:
        with open(kd) as f:
            knowledge_doc_texts[".".join(Path(kd).name.split(".")[:-1])] = f.read()
    documents = []
    metadatas = []
    ids = []
    for kd, doc in knowledge_doc_texts.items():
        _logger.info(f"Splitting knowledge document: {kd}")
        for idx, passage in enumerate(window_document(kd, doc)):
            documents.append(passage)
            metadatas.append({"file": kd, "part": idx})
            ids.append(kd + f"_{idx}")

    knowledge_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="HSSLoader",
        description="Stores datajunction node data and knowledge documents into a chroma vector database",
        epilog="https://github.com/KnowledgeForge/headjack",
    )
    parser.add_argument(
        "--knowledge",
        help="A glob pattern matching knowledge text files",
        type=str,
    )
    parser.add_argument("--dj", help="The DataJunction URL", type=str)
    parser.add_argument("--chroma-host", help="The Chroma DB Host", type=str)
    parser.add_argument("--chroma-port", help="The Chroma DB Port", type=str)
    args = parser.parse_args()

    client = get_chroma_client(host=args.chroma_host, port=args.chroma_port)
    skipped_metrics = False
    skipped_knowledge = False
    try:
        client.get_collection(name="metrics")
        _logger.info("Skipped embedding metrics data, collection already exists.")
        skipped_metrics = True
    except Exception:
        index_metrics(dj_url=args.dj, client=client)
        _logger.info("Metrics indexing completed")

    try:
        client.get_collection(name="knowledge")
        _logger.info("Skipped embedding knowledge data, collection already exists.")
        skipped_knowledge = True
    except Exception:
        index_knowledge(knowledge_dir=args.knowledge, client=client)
        _logger.info("Knowledge indexing completed")

    if skipped_metrics:
        sys.exit("`metrics` collection already exists")

    if skipped_knowledge:
        sys.exit("`knowledge` collection already exists")
