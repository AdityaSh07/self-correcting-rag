from typing import Literal, TypedDict, List, Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    chat_history: Annotated[list[AnyMessage], add_messages]
    question: str

    # what we actually send to vector retriever
    retrieval_query: str
    rewrite_tries: int
    
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str

    # Post-generation verification
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]

    retries: int

    isuse: Literal["useful", "not_useful"]
    use_reason: str

    chat_history: List[BaseMessage]
