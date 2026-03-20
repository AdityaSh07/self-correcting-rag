from typing import Literal, TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .state import State
from .llm import llm
from .prompts import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
    isuse_prompt, 
    rewrite_for_retrieval_prompt

)
from .schemas import RetrieveDecision, RelevanceDecision, IsSUPDecision, IsUSEDecision, RewriteDecision
from .graders import should_retrieve_llm, relevance_llm, issup_llm, isuse_llm, rewrite_llm
from .retriever import retriever

#---------------------------------------------------------------------------------------------------
MAX_RETRIES = 3
#---------------------------------------------------------------------------------------------------

def decide_retrieval(state: State):
    chat_history = state.get("chat_history", [])
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(
            question=state["question"],
            chat_history=chat_history
        )
    )
    return {
        "need_retrieval": decision.should_retrieve
    }

def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"

def generate_direct(state: State):
    chat_history = state.get("chat_history", [])
    out = llm.invoke(direct_generation_prompt.format_messages(
        question=state["question"],
        chat_history=chat_history
    ))
    return {"answer": out.content}

def retrieve(state: State):
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}


def is_relevant(state: State):
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


def generate_from_context(state: State):
    context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
    if not context:
        return {"answer": "No answer found.", "context": ""}
    
    chat_history = state.get("chat_history", [])
    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"], 
            context=context,
            chat_history=chat_history
        )
    )
    return {"answer": out.content, "context": context}

def no_answer_found(state: State):
    return {"answer": "No answer found.", "context": ""}

def update_history(state: State):
    return {
        "chat_history": [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["answer"])
        ]
    }

def is_sup(state: State):
    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}




def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    # fully supported -> move forward to IsUSE (via "accept_answer" label)
    if state.get("issup") == "fully_supported":
        return "accept_answer"

    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"  # will go to is_use, then likely not_useful -> no_answer_found

    return "revise_answer"

def accept_answer(state: State):
    return {}  # keep answer as-is


def revise_answer(state: State):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,  # ✅ increment
    }


def is_use(state: State):
    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}

MAX_REWRITE_TRIES = 3  # tune (2–4 is usually fine)

def route_after_isuse(state: State) -> Literal["end_with_updated_history", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "end_with_updated_history"  # will route to update_history -> END

    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"

    return "rewrite_question"

def rewrite_question(state: State):
    chat_history = state.get("chat_history", [])
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
            chat_history=chat_history
        )
    )

    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # ✅ optional: reset these so next pass is clean
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }



