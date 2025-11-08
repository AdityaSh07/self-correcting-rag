# 🤖 Self-Correcting RAG: An Adaptive LLM Agent
This project is an advanced Retrieval-Augmented Generation (RAG) chatbot built with LangGraph. It goes far beyond a simple RAG chain by implementing a stateful, cyclic graph that allows the agent to iteratively self-correct its own retrieval and generation processes.

The system can reason about the quality of its own work. It dynamically routes its flow to:

Grade retrieved documents for relevance.

Transform the user's query if retrieval fails.

Grade its own generated answer for hallucinations.

Loop for a set number of retries before gracefully failing.

Provide a fallback answer from the LLM's general knowledge as a last resort.

This repository serves as a production-ready pattern for building reliable and fault-tolerant LLM applications.

# ✨ Core Features
Adaptive RAG Flow: Uses a LangGraph state machine to dynamically route based on the quality of retrieved context and generated answers.

Iterative Self-Correction: The graph can loop back on itself. If a generated answer is found to be ungrounded, the system automatically triggers a new attempt by transforming the query.

Document Relevance Grading: A dedicated grading_documents node filters out irrelevant context before it reaches the generator, saving compute and improving accuracy.

Robust Error Handling:

Max Iteration Limit: A built-in counter (check_iteration) prevents infinite loops.

Fallback Synthesis: If the system cannot find a grounded answer after its retries, it routes to a generate_fallback_answer node, using the LLM's general knowledge while issuing a clear warning to the user.

Input Validation: A fast-fail path (query_not_at_all_relevant) rejects nonsensical or out-of-scope queries immediately.

Full-Stack Interface: A Flask API serves the RAG agent, and a custom HTML/CSS/JavaScript frontend provides a simple chat interface.

# ⚙️ System Architecture
The core of this project is the LangGraph state machine. The flow is not linear; it's a cyclic graph that routes based on a series of LLM-powered checks.

The Flow Explained:
Retrieve (START): Fetches context from ChromaDB.

Grade Documents: Checks if the retrieved docs are relevant.

If YES: Proceeds to content_generator.

If NO: Routes to check_iteration to begin a retry.

Generate: Creates an answer based on the (now-verified) context.

Grade Generation: Checks the answer for hallucinations and relevance.

If YES ("useful"): Routes to END and returns the answer.

If NO ("not useful"): Routes to check_iteration.

The Loop Gate (Error Handling):

check_iteration: Increments the retry counter.

route: A router checks the counter against max_count.

If Retries Left: Routes to transform_user_query.

If Max Retries Hit: Routes to generate_fallback_answer (Graceful Exit 1).

The Retry Path:

transform_user_query: Rewrites the query.

If Query is Irrelevant: Routes to END (Graceful Exit 2).

If Query is Rewritten: Routes back to vector_retrieved_docs to restart the loop.

![RAG Workflow Architecture](assets/workflow.png)

# 🛠️ Tech Stack
Frameworks: LangChain, LangGraph

LLM & API: Google Gemini 2.5 Flash

Embedding Model: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (is running locally in this project)

Vector Store: Chroma DB

Backend: Flask

Frontend: HTML/CSS/JavaScript

# Dependencies:
langchain-google-genai langchain-huggingface langgraph langchain-core langchain-community python-dotenv Flask