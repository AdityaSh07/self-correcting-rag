# Autonomous Self-Correcting RAG Agent: A Production-Ready AI

This project is a highly sophisticated, stateful Retrieval-Augmented Generation (RAG) chatbot, engineered with LangGraph and FastAPI. It transcends the limitations of traditional, linear LLM queries by acting as an agentic system. It utilizes a cyclic state graph to actively monitor, reflect on, and correct its own retrieval and generative processes in real-time.

By incorporating dedicated nodes to evaluate **document relevance**, **detect hallucinations**, **verify groundings**, and **autonomously rewrite queries**, this architecture ensures high-fidelity, contextually accurate answers to user inquiries.

## Why LangGraph Excels Over Traditional RAG and Standard API Calls

Traditional RAG and simple LLM API calls execute a linear "one-shot" process: searching a vector database once, feeding the result to an LLM, and outputting the response. If the retrieved chunks miss the mark or lack the complete context, the system risks outputting a hallucination or failing entirely.

How this LangGraph architecture differs:
*   **Self-Reflection and Evaluation**: Before serving an answer to the user, the agent evaluates it against the retrieved context (IsSUP) to ensure it is fully grounded. It also verifies if the answer genuinely addresses the user's intent (IsUSE). 
*   **Autonomous Query Rewriting**: If initial retrieved results are poor, or if an answer is deemed unhelpful, the system automatically rewrites the user's question, optimizes it for vector search (resolving pronouns based on chat history), and attempts retrieval again without requiring explicit user intervention.
*   **Stateful Memory Systems**: Unlike stateless API calls, this system tracks the conversation graph securely using LangGraph's checkpointers, persisting conversation threads organically to handle follow-up questions accurately.
*   **Early Exit Routing Strategy**: Computations and API calls are conserved by an initial routing node (`decide_retrieval`). Simple greetings or questions answerable from existing chat history bypass the expensive vector retrieval steps entirely.

## Core System Capabilities

*   **Adaptive Routing**: Dynamically chooses between direct LLM generation and vector retrieval via FAISS based on the classification of the query.
*   **Iterative Self-Correction Loop**: Answers found to be ungrounded or unsupported immediately trigger a query transformation and re-retrieval sequence.
*   **Document Relevance Grading**: A dedicated evaluation node strictly filters document noise and irrelevant context before passing it to the generator.
*   **Robust Fallback and Retry Logic**: The system inherently limits itself to prevent infinite loops. Once all automated interventions fail, the agent defaults to a graceful failure handle.
*   **Streaming Responses**: Uses LangGraph's token-by-token message streaming by intersecting graph execution with FastAPI to stream final, verified answers to users in real-time.
*   **Secure API Architecture**: Built entirely on FastAPI, the system incorporates robust session management utilizing stateless JWTs delivered exclusively via `HttpOnly`, `SameSite=Lax` cookies, neutralizing XSS vulnerabilities and securing token transport for authenticated interactions.

## System Architecture and Flow

This cyclic agent workflow routes paths using LangChain deterministic decision nodes:

1.  Intent Check (`decide_retrieval`) -> Determine if document retrieval is needed or if general knowledge/conversation history suffices.
2.  Retrieve (`vector_retrieved_docs`) -> Extract similar topical chunks using FAISS.
3.  Grade Relevance (`is_relevant`) -> Evaluate if the chunks are semantically on-topic to the user's inquiry.
4.  Generative Output (`content_generator`) -> Formulate the answer strictly from verified context limits.
5.  Grounding Verification (`is_sup`) -> Detect potential hallucinations (Self-Correction Trigger).
6.  Usefulness Verification (`is_use`) -> Ensure the original question intent was explicitly satisfied by the constructed answer.
7.  Auto-Transformation (`rewrite_question`) -> The query is abstractly rewritten to extract improved vector similarity from FAISS if the workflow loop is required to iterate further.

![LangGraph Agent Workflow Architecture](assets/rag_graph.png)

## Technology Stack

*   Agent Framework: LangChain & LangGraph
*   LLM Inference: Groq (via langchain-groq for high-speed inference)
*   Vector Store: FAISS (Facebook AI Similarity Search) running locally
*   Embeddings: HuggingFace sentence-transformers/all-MiniLM-L6-v2
*   Backend REST API: FastAPI & Uvicorn
*   Database: PostgreSQL with SQLAlchemy (Thread/User Persistence)
*   Frontend Interface: Custom HTML, CSS, and JavaScript   