import logging
import threading

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import LLMConfig
from .llm import embed_model

logger = logging.getLogger(__name__)

# --------------------------------- CONSTANTS ----------------------------------

DOCUMENTS_DIR = LLMConfig.DOCUMENTS_DIR
DOCUMENTS_GLOB = LLMConfig.DOCUMENTS_GLOB
CHROMA_PERSIST_DIR = LLMConfig.CHROMA_PERSIST_DIR
CHROMA_COLLECTION = LLMConfig.CHROMA_COLLECTION

CHUNK_SIZE = LLMConfig.CHUNK_SIZE
CHUNK_OVERLAP = LLMConfig.CHUNK_OVERLAP
RETRIEVER_SEARCH_TYPE = LLMConfig.RETRIEVER_SEARCH_TYPE
RETRIEVER_K = LLMConfig.RETRIEVER_K
RETRIEVER_LAMBDA_MULT = LLMConfig.RETRIEVER_LAMBDA_MULT


# --------------------------------- BUILD -------------------------------------


def load_and_split_documents():
    loader = DirectoryLoader(
        path=DOCUMENTS_DIR,
        glob=DOCUMENTS_GLOB,
        loader_cls=TextLoader,
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def build_vector_store(chunks):
    vec_store = Chroma(
        embedding_function=embed_model,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
    )
    vec_store.add_documents(chunks)
    return vec_store

def _build_retriever():
    """Build the document retriever from scratch."""
    chunks = load_and_split_documents()
    vec_store = build_vector_store(chunks)
    retriever = vec_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVER_K, "lambda_mult": RETRIEVER_LAMBDA_MULT},
    )
    return retriever


# --------------------------------- SINGLETON ---------------------------------

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
                return None
    return _retriever_instance


# --------------------------------- HELPERS -----------------------------------


def format_docs(retrieved_docs) -> str:
    """Concatenate retrieved document contents separated by double newlines."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)
