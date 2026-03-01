"""All node and conditional-edge functions for the RAG LangGraph."""

import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ...prompts.prompts import fallback_system_prompt
from .graders import (
    grade_prompt,
    structured_relevance_grader,
    hallucination_prompt,
    structured_hallucination_grader,
    answer_prompt,
    structured_answer_grader,
    question_rewriter,
)
from .llm import rag_chain, model
from .retriever import get_or_create_retriever
from .state import GraphState

logger = logging.getLogger(__name__)


# --------------------------------- NODE FUNCTIONS ----------------------------


def retrieve_context(state: GraphState) -> dict:
    """Retrieve relevant documents from the vector store for the given question."""
    question = state["question"]
    retriever = get_or_create_retriever()

    if retriever is None:
        logger.warning("Retriever unavailable — returning empty documents")
        return {"documents": [], "question": question, "chat_history": [HumanMessage(question)]}

    try:
        documents = retriever.invoke(question)
    except Exception:
        logger.exception("Retriever invocation failed for question: %s", question)
        documents = []

    return {"documents": documents, "question": question, "chat_history": [HumanMessage(question)]}


def grade_documents(state: GraphState) -> dict:
    """Grade each retrieved document for relevance to the user question."""
    logger.info("Checking document relevance to question")
    question = state["question"]
    documents = state["documents"]

    filtered_docs: list = []
    unfiltered_docs: list = []

    for doc in documents:
        try:
            prompt = grade_prompt.invoke({"question": question, "document": doc})
            score = structured_relevance_grader.invoke(prompt)
            grade = score.binary_score
        except Exception:
            logger.exception("Relevance grading failed for a document — treating as irrelevant")
            grade = "no"

        if grade == "yes":
            logger.debug("Document RELEVANT")
            filtered_docs.append(doc)
        else:
            logger.debug("Document NOT RELEVANT")
            unfiltered_docs.append(doc)

    if len(unfiltered_docs) > 1:
        return {"unfilter_documents": unfiltered_docs, "filter_documents": [], "question": question}
    return {"filter_documents": filtered_docs, "unfilter_documents": [], "question": question}


def generate(state: GraphState) -> dict:
    """Generate an answer using the RAG chain given question and documents."""
    logger.info("Generating answer")
    question = state["question"]
    documents = state["documents"]

    try:
        generation = rag_chain.invoke({"context": documents, "question": question})
    except Exception:
        logger.exception("RAG chain generation failed")
        return {
            "documents": documents,
            "question": question,
            "generation": "An error occurred while generating the answer. Please try again.",
            "chat_history": [AIMessage("An error occurred while generating the answer.")],
        }

    return {
        "documents": documents,
        "question": question,
        "generation": generation.content,
        "chat_history": [AIMessage(generation.content)],
    }


def transform_query(state: GraphState) -> dict:
    """Rewrite the user question for better retrieval, or flag it as irrelevant."""
    question = state["question"]
    documents = state["documents"]

    logger.debug("Rewriting query — current documents: %d", len(documents))

    try:
        response = question_rewriter.invoke({"question": question, "documents": documents})
    except Exception:
        logger.exception("Query rewriting failed")
        return {"documents": documents, "question": question}

    logger.info("Rewritten query: %s", response)

    if response == "question not relevant":
        logger.info("Question deemed entirely irrelevant")
        return {"documents": documents, "question": response, "generation": "question was not at all relevant"}
    return {"documents": documents, "question": response}


def check_iteration(state: GraphState) -> dict:
    """Increment the retry counter."""
    count = state["count"]
    logger.debug("Iteration count incremented to %d", count + 1)
    return {"count": count + 1}


def generate_fallback_answer(state: GraphState) -> dict:
    """Generate an unverified answer when the RAG pipeline exhausts retries."""
    logger.warning("Fallback mode: max retries hit — generating unverified answer")
    question = state["question"]
    chat_history = state["chat_history"]

    fallback_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", fallback_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Final Question for General Knowledge Answer: {question}"),
        ]
    )
    fallback_chain = fallback_prompt_template | model | StrOutputParser()

    try:
        final_answer = fallback_chain.invoke({"question": question, "chat_history": chat_history})
    except Exception:
        logger.exception("Fallback generation failed")
        final_answer = "I'm sorry, I was unable to generate an answer at this time. Please try again later."

    return {
        "generation": final_answer,
        "chat_history": [AIMessage(final_answer)],
    }


# --------------------------------- CONDITIONAL EDGES -------------------------


def decide_to_generate(state: GraphState) -> str:
    """Decide whether to generate an answer or re-transform the query."""
    logger.info("Assessing graded documents")
    unfiltered_documents = state["unfilter_documents"]
    filtered_documents = state["filter_documents"]

    if unfiltered_documents:
        logger.info("Documents not relevant — will transform query")
        return "check_iter"
    if filtered_documents:
        logger.info("Relevant documents found — will generate")
        return "generate"
    logger.warning("No documents available after grading — will transform query")
    return "check_iter"


def decide_to_generate_after_transformation(state: GraphState) -> str:
    """Route after query transformation: retry retrieval or end."""
    if state["question"] == "question not relevant":
        return "query_not_at_all_relevant"
    return "Retriever"


def grade_generation_vs_documents_and_question(state: GraphState) -> str:
    """Grade the generation for hallucinations and question relevance."""
    logger.info("Checking for hallucinations")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    try:
        prompt = hallucination_prompt.invoke({"documents": documents, "generation": generation})
        score = structured_hallucination_grader.invoke(prompt)
        grade = score.binary_score
    except Exception:
        logger.exception("Hallucination grading failed — treating as not useful")
        return "not useful"

    if grade == "yes":
        logger.info("Generation is grounded in documents")

        try:
            ans_prompt = answer_prompt.invoke({"question": question, "generation": generation})
            score = structured_answer_grader.invoke(ans_prompt)
            grade = score.binary_score
        except Exception:
            logger.exception("Answer grading failed — treating as not useful")
            return "not useful"

        if grade == "yes":
            logger.info("Generation addresses the question")
            return "useful"
        else:
            logger.info("Generation does not address the question — retrying")
            return "not useful"
    else:
        logger.info("Generation is NOT grounded in documents — retrying")
        return "not useful"


def route(state: GraphState) -> str:
    """Route based on whether the max retry count has been reached."""
    if state["count"] == state["max_count"]:
        logger.warning("Max retry count (%d) reached", state["max_count"])
        return "max_count_reached_end"
    return "transform_user_query"
