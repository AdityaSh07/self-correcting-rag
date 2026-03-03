import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from langchain_core.messages import AIMessageChunk

from .. import oauth2, database, models, schemas
from ..rag import rag_chatbot, GraphState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"],
)

# Only stream tokens produced by these graph nodes (the actual answer generators).
# Other nodes that call the LLM (grading, rewriting) are excluded.
_STREAM_NODES = frozenset({"content_generator", "generate_fallback_answer"})


async def stream_rag_response(question: str, user_id: int):
    """Stream LLM tokens from the RAG graph as they are generated.

    Uses LangGraph's ``astream(stream_mode="messages")`` which intercepts
    every chat-model call inside the graph and yields ``AIMessageChunk``
    objects in real time, while the nodes still receive the full response
    for state updates.  We filter chunks so only the final-answer nodes
    (``content_generator`` and ``generate_fallback_answer``) are forwarded
    to the client.
    """
    initial_state: GraphState = {
        "question": question,
        "chat_history": [],
        "generation": "",
        "generation_grade": "",
        "documents": [],
        "filter_documents": [],
        "unfilter_documents": [],
        "count": 0,
        "max_count": 3,
    }

    config = {"configurable": {"thread_id": f"user_{user_id}"}}

    # Buffer chunks per generation node so we only yield the LAST one.
    # This prevents intermediate RAG retry answers from leaking to the user
    # when the pipeline eventually falls back to generate_fallback_answer.
    node_chunks: dict[str, list[str]] = {}
    last_generation_node: str | None = None

    try:
        async for msg_chunk, metadata in rag_chatbot.astream(
            initial_state, config, stream_mode="messages"
        ):
            node = metadata.get("langgraph_node")
            if (
                isinstance(msg_chunk, AIMessageChunk)
                and node in _STREAM_NODES
                and msg_chunk.content
            ):
                node_chunks.setdefault(node, []).append(msg_chunk.content)
                last_generation_node = node

        if last_generation_node:
            # Yield only the final generation node's buffered output
            for chunk in node_chunks[last_generation_node]:
                yield chunk
        else:
            # Graph ended without any generation node producing output
            # (e.g. query deemed entirely irrelevant — routed to END directly)
            state_snapshot = rag_chatbot.get_state(config)
            generation = state_snapshot.values.get("generation", "")
            yield generation or (
                "I couldn't generate a response. "
                "Please try rephrasing your question."
            )

    except Exception as exc:
        logger.exception("RAG streaming error")
        yield f"Error processing your request: {exc}"


@router.post("/stream")
async def chat_stream(
    request: schemas.ChatRequest,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user),
):
    """Stream responses from the RAG chatbot token-by-token."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return StreamingResponse(
        stream_rag_response(request.message, current_user.id),
        media_type="text/plain",
    )

