from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import (
    retrieve_context,
    grade_documents,
    generate,
    transform_query,
    check_iteration,
    generate_fallback_answer,
    grade_generation,
    decide_to_generate,
    decide_to_generate_after_transformation,
    grade_generation_vs_documents_and_question,
    route,
)

# --------------------------------- GRAPH NODES & EDGES (structure) -------------------------

graph = StateGraph(GraphState)

graph.add_node("vector_retrieved_docs", retrieve_context)
graph.add_node("grading_documents", grade_documents)
graph.add_node("content_generator", generate)
graph.add_node("grade_generation", grade_generation)
graph.add_node("transform_user_query", transform_query)
graph.add_node("check_iteration", check_iteration)
graph.add_node("generate_fallback_answer", generate_fallback_answer)

# --- EDGES ---

# 1. Start -> Retrieve
graph.add_edge(START, "vector_retrieved_docs")

# 2. Retrieve -> Grade Docs
graph.add_edge("vector_retrieved_docs", "grading_documents")

# 3. Grade Docs -> Decide Path
graph.add_conditional_edges(
    "grading_documents",
    decide_to_generate,
    {"generate": "content_generator", "check_iter": "check_iteration"},
)

# 4. Content Generator -> Grade Generation (own node so grading LLM calls don't stream)
graph.add_edge("content_generator", "grade_generation")

# 5. Grade Generation -> Decide Path (pure state reader, no LLM calls)
graph.add_conditional_edges(
    "grade_generation",
    grade_generation_vs_documents_and_question,
    {"useful": END, "not useful": "check_iteration"},
)

# 6. Check Iteration -> Decide Loop Fate
graph.add_conditional_edges(
    "check_iteration",
    route,
    {"transform_user_query": "transform_user_query", "max_count_reached_end": "generate_fallback_answer"},
)

# 7. Transform Query -> Re-retrieve or End
graph.add_conditional_edges(
    "transform_user_query",
    decide_to_generate_after_transformation,
    {"Retriever": "vector_retrieved_docs", "query_not_at_all_relevant": END},
)

# 8. Fallback -> End
graph.add_edge("generate_fallback_answer", END)

# --------------------------------- COMPILE ------------------------------------

checkpointer = InMemorySaver()
rag_chatbot = graph.compile(checkpointer=checkpointer)
