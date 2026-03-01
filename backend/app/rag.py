import logging
import threading
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ..prompts.prompts import (
    system_text_hallucination,
    system_text_relevance,
    system_text_answer_eval,
    system,
    fallback_system_prompt,
)

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------- CONSTANTS ----------------------------------

LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.8
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DOCUMENTS_DIR = "documents"
DOCUMENTS_GLOB = "*.txt"
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION = "embeddings"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_SEARCH_TYPE = "mmr"
RETRIEVER_K = 5
RETRIEVER_LAMBDA_MULT = 0.8

RAG_PROMPT_HUB_ID = "rlm/rag-prompt"

# --------------------------------- MODELS -------------------------------------

model = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

rag_prompt = hub.pull(RAG_PROMPT_HUB_ID)
rag_chain = rag_prompt | model


# --------------------------------- RETRIEVER ----------------------------------

def _build_retriever():
    """Build the document retriever.

    Loads text documents from *DOCUMENTS_DIR*, splits them into chunks,
    and indexes them in a Chroma vector store. Documents are only added
    when the collection is empty, preventing duplicates on app restarts.
    """
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

    vec_store = Chroma(
        embedding_function=embed_model,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
    )

    # Only add documents when the collection is empty to prevent duplicates
    # accumulating across app restarts.
    if vec_store._collection.count() == 0:
        logger.info("Indexing %d chunks into vector store", len(chunks))
        vec_store.add_documents(chunks)
    else:
        logger.info("Reusing existing index (%d docs)", vec_store._collection.count())

    retriever = vec_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVER_K, "lambda_mult": RETRIEVER_LAMBDA_MULT},
    )
    return retriever


# Thread-safe lazy initialisation
_retriever_lock = threading.Lock()
_retriever_instance = None


def get_or_create_retriever():
    """Return the singleton retriever, creating it on first call (thread-safe)."""
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


def format_docs(retrieved_docs) -> str:
    """Concatenate retrieved document contents separated by double newlines."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# --------------------------------- GRADING SCHEMAS ----------------------------


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_relevance_grader = model.with_structured_output(GradeDocuments)
structured_hallucination_grader = model.with_structured_output(GradeHallucinations)
structured_answer_grader = model.with_structured_output(GradeAnswer)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_text_relevance),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_text_hallucination),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_text_answer_eval),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)


# --------------------------------- QUERY REWRITER ----------------------------

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n,"
            " Here is the document: \n\n {documents} \n ,"
            " Formulate an improved question. if possible other return 'question not relevant'.",
        ),
    ]
)
question_rewriter = re_write_prompt | model | StrOutputParser()


# --------------------------------- GRAPH STATE --------------------------------


class GraphState(TypedDict):
    """Shared state flowing through every node of the RAG graph."""

    chat_history: Annotated[list[BaseMessage], add_messages]
    question: str
    generation: str
    documents: list[str]
    filter_documents: list[str]
    unfilter_documents: list[str]
    count: int
    max_count: int


# --------------------------------- NODE FUNCTIONS -----------------------------


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
    # Fallback: no documents at all
    logger.warning("No documents available after grading — will transform query")
    return "check_iter"


def generate(state: GraphState) -> dict:
    """Generate an answer using the RAG chain given question and documents."""
    logger.info("Generating answer")
    question = state["question"]
    documents = state["documents"]

    try:
        generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
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


def check_iteration(state: GraphState) -> dict:
    """Increment the retry counter."""
    count = state["count"]
    logger.debug("Iteration count incremented to %d", count + 1)
    return {"count": count + 1}


def route(state: GraphState) -> str:
    """Route based on whether the max retry count has been reached."""
    if state["count"] == state["max_count"]:
        logger.warning("Max retry count (%d) reached", state["max_count"])
        return "max_count_reached_end"
    return "transform_user_query"


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


# --------------------------------- GRAPH CONSTRUCTION -------------------------

graph = StateGraph(GraphState)
graph.add_node("vector_retrieved_docs", retrieve_context)
graph.add_node("grading_documents", grade_documents)
graph.add_node("content_generator", generate)
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

# 4. Content Generator -> Decide Path
graph.add_conditional_edges(
    "content_generator",
    grade_generation_vs_documents_and_question,
    {"useful": END, "not useful": "check_iteration"},
)

# 5. Check Iteration -> Decide Loop Fate
graph.add_conditional_edges(
    "check_iteration",
    route,
    {"transform_user_query": "transform_user_query", "max_count_reached_end": "generate_fallback_answer"},
)

# 6. Transform Query -> Re-retrieve or End
graph.add_conditional_edges(
    "transform_user_query",
    decide_to_generate_after_transformation,
    {"Retriever": "vector_retrieved_docs", "query_not_at_all_relevant": END},
)

# 7. Fallback -> End
graph.add_edge("generate_fallback_answer", END)

# --------------------------------- COMPILE ------------------------------------

checkpointer = InMemorySaver()
rag_chatbot = graph.compile(checkpointer=checkpointer)