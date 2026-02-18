from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio

from .. import oauth2, database, models, schemas
from ..rag import rag_chatbot, GraphState
from langchain_core.messages import HumanMessage


router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"],
)


async def stream_rag_response(question: str, user_id: int):
    """
    Invoke the RAG graph and stream the final answer.
    """
    try:
        # Initialize the graph state
        initial_state: GraphState = {
            "question": question,
            "chat_history": [],
            "generation": "",
            "documents": [],
            "filter_documents": [],
            "unfilter_documents": [],
            "count": 0,
            "max_count": 3,
        }
        
        # Run the RAG graph with a config (unique thread_id per user for checkpointing)
        # Use asyncio.to_thread to run the synchronous invoke in a thread pool
        config = {"configurable": {"thread_id": f"user_{user_id}"}}
        final_state = await asyncio.to_thread(rag_chatbot.invoke, initial_state, config)
        
        # Extract the final generation
        final_answer = final_state.get("generation", "")
        
        if not final_answer:
            final_answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        # Stream the answer word by word for better UX
        words = final_answer.split()
        for i, word in enumerate(words):
            if i > 0:
                yield " "
            yield word
            # Small delay to simulate streaming (can be removed for faster response)
            await asyncio.sleep(0.02)
            
    except Exception as e:
        import traceback
        error_msg = f"Error processing your request: {str(e)}\n"
        # Log the full traceback for debugging
        print(f"RAG Error: {traceback.format_exc()}")
        for char in error_msg:
            yield char
            await asyncio.sleep(0.01)


@router.post("/stream")
async def chat_stream(
    request: schemas.ChatRequest,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user),
):
    """
    Stream responses from the RAG chatbot.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    async def event_generator():
        async for chunk in stream_rag_response(request.message, current_user.id):
            yield chunk
    
    return StreamingResponse(event_generator(), media_type="text/plain")

