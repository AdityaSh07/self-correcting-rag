import logging
import threading
import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import llm_config as LLMConfig
from .llm import embed_model

logger = logging.getLogger(__name__)

# --------------------------------- CONSTANTS ----------------------------------

DOCUMENTS_DIR = LLMConfig.DOCUMENTS_DIR
DOCUMENTS_GLOB = LLMConfig.DOCUMENTS_GLOB
FAISS_PERSIST_DIR = LLMConfig.FAISS_PERSIST_DIR
FAISS_COLLECTION = LLMConfig.FAISS_COLLECTION

CHUNK_SIZE = LLMConfig.CHUNK_SIZE
CHUNK_OVERLAP = LLMConfig.CHUNK_OVERLAP
RETRIEVER_SEARCH_TYPE = LLMConfig.RETRIEVER_SEARCH_TYPE
RETRIEVER_K = LLMConfig.RETRIEVER_K
RETRIEVER_LAMBDA_MULT = LLMConfig.RETRIEVER_LAMBDA_MULT

# --------------------------------- MAIN --------------------------------------

def load_and_split_documents():
    loader = DirectoryLoader(str(DOCUMENTS_DIR), glob=DOCUMENTS_GLOB, loader_cls=PyPDFLoader)
    docs = loader.load()
    
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

    return chunks

def _build_retriever():
    """Build the document retriever from scratch."""
    chunks = load_and_split_documents()
    vec_store = FAISS.from_documents(chunks, embed_model)
    retriever = vec_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVER_K, "lambda_mult": RETRIEVER_LAMBDA_MULT},
    )
    return retriever


_retriever_lock = threading.Lock()
_retriever_instance = None


def get_or_create_retriever():
    """Ensure only one retriever instance is created, even if multiple threads call this simultaneously."""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance
    with _retriever_lock:
        # Double-checked locking
        if _retriever_instance is None:
            try:
                _retriever_instance = _build_retriever()
            except Exception:
                logger.exception("Failed to initialise retriever")
    return _retriever_instance

# Create a singleton retriever instance to be imported by nodes.py
retriever = get_or_create_retriever()
