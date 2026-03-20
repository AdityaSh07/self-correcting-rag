from langgraph.graph import StateGraph, START, END
from .state import State
from .nodes import *
from langgraph.checkpoint.memory import InMemorySaver

g = StateGraph(State)

# --------------------
# Nodes
# --------------------
g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)

g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_answer_found", no_answer_found)


g.add_node("is_sup", is_sup)
g.add_node("revise_answer", revise_answer)


g.add_node("is_use", is_use)


g.add_node("rewrite_question", rewrite_question)

g.add_node("update_history", update_history)

# --------------------
# Edges
# --------------------
g.add_edge(START, "decide_retrieval")

g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {"generate_direct": "generate_direct", "retrieve": "retrieve"},
)

g.add_edge("generate_direct", "update_history")


g.add_edge("retrieve", "is_relevant")

g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found",
    },
)

g.add_edge("no_answer_found", "update_history")


g.add_edge("generate_from_context", "is_sup")

g.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {
        "accept_answer": "is_use",      # fully_supported (or max retries) -> go to IsUSE
        "revise_answer": "revise_answer",
    },
)

g.add_edge("revise_answer", "is_sup")  


g.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "end_with_updated_history": "update_history",
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found",
    },
)

g.add_edge("update_history", END) # 


g.add_edge("rewrite_question", "retrieve")


checkpointer = InMemorySaver()
rag_chatbot = g.compile(checkpointer=checkpointer)
rag_chatbot
