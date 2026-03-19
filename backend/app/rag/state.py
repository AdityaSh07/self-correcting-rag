from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):

    

    chat_history: Annotated[list[BaseMessage], add_messages]

    question: str

    generation: str

    documents: list[str]

    filter_documents: list[str]

    unfilter_documents: list[str]

    count: int

    max_count: int
