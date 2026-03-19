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


# # --------------------------------- NODE FUNCTIONS ----------------------------


# def retrieve_context(state: GraphState) -> dict:
#     """Retrieve relevant documents from the vector store for the given question."""
#     question = state["question"]
#     retriever = get_or_create_retriever()

#     if retriever is None:
#         logger.warning("Retriever unavailable — returning empty documents")
#         return {"documents": [], "question": question, "chat_history": [HumanMessage(question)]}

#     try:
#         documents = retriever.invoke(question)
#     except Exception:
#         logger.exception("Retriever invocation failed for question: %s", question)
#         documents = []

#     return {"documents": documents, "question": question, "chat_history": [HumanMessage(question)]}


# def grade_documents(state: GraphState) -> dict:
#     """Grade each retrieved document for relevance to the user question."""
#     logger.info("Checking document relevance to question")
#     question = state["question"]
#     documents = state["documents"]

#     filtered_docs: list = []
#     unfiltered_docs: list = []

#     for doc in documents:
#         try:
#             prompt = grade_prompt.invoke({"question": question, "document": doc})
#             score = structured_relevance_grader.invoke(prompt)
#             grade = score.binary_score.strip().lower()
#         except Exception:
#             logger.exception("Relevance grading failed for a document — treating as irrelevant")
#             grade = "no"

#         if grade == "yes":
#             logger.debug("Document RELEVANT")
#             filtered_docs.append(doc)
#         else:
#             logger.debug("Document NOT RELEVANT")
#             unfiltered_docs.append(doc)

#     if not filtered_docs:
#         logger.info("No relevant documents found — all %d doc(s) were irrelevant", len(unfiltered_docs))
#         return {"documents": [], "unfilter_documents": unfiltered_docs, "filter_documents": [], "question": question}
#     return {"documents": filtered_docs, "filter_documents": filtered_docs, "unfilter_documents": [], "question": question}


# def generate(state: GraphState) -> dict:
#     """Generate an answer using the RAG chain given question and documents."""
#     logger.info("Generating answer")
#     question = state["question"]
#     documents = state["documents"]

#     try:
#         generation = rag_chain.invoke({"context": documents, "question": question})
#     except Exception:
#         logger.exception("RAG chain generation failed")
#         return {
#             "documents": documents,
#             "question": question,
#             "generation": "An error occurred while generating the answer. Please try again.",
#             "chat_history": [AIMessage("An error occurred while generating the answer.")],
#         }

#     return {
#         "documents": documents,
#         "question": question,
#         "generation": generation.content,
#         "chat_history": [AIMessage(generation.content)],
#     }


# def transform_query(state: GraphState) -> dict:
#     """Rewrite the user question for better retrieval, or flag it as irrelevant."""
#     question = state["question"]
#     documents = state["documents"]

#     logger.debug("Rewriting query — current documents: %d", len(documents))

#     try:
#         response = question_rewriter.invoke({"question": question, "documents": documents})
#     except Exception:
#         logger.exception("Query rewriting failed")
#         return {"documents": documents, "question": question}

#     logger.info("Rewritten query: %s", response)

#     if response == "question not relevant":
#         logger.info("Question deemed entirely irrelevant")
#         return {"documents": documents, "question": response, "generation": "question was not at all relevant"}
#     return {"documents": documents, "question": response}


# def check_iteration(state: GraphState) -> dict:
#     """Increment the retry counter."""
#     count = state["count"]
#     logger.debug("Iteration count incremented to %d", count + 1)
#     return {"count": count + 1}


# def generate_fallback_answer(state: GraphState) -> dict:
#     """Generate an unverified answer when the RAG pipeline exhausts retries."""
#     logger.warning("Fallback mode: max retries hit — generating unverified answer")
#     question = state["question"]
#     chat_history = state["chat_history"]

#     fallback_prompt_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", fallback_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "Final Question for General Knowledge Answer: {question}"),
#         ]
#     )
#     fallback_chain = fallback_prompt_template | model | StrOutputParser()

#     try:
#         final_answer = fallback_chain.invoke({"question": question, "chat_history": chat_history})
#     except Exception:
#         logger.exception("Fallback generation failed")
#         final_answer = "I'm sorry, I was unable to generate an answer at this time. Please try again later."

#     return {
#         "generation": final_answer,
#         "chat_history": [AIMessage(final_answer)],
#     }


# # --------------------------------- CONDITIONAL EDGES -------------------------


# def decide_to_generate(state: GraphState) -> str:
#     """Decide whether to generate an answer or re-transform the query."""
#     logger.info("Assessing graded documents")
#     unfiltered_documents = state["unfilter_documents"]
#     filtered_documents = state["filter_documents"]

#     if filtered_documents:
#         logger.info("Relevant documents found — will generate")
#         return "generate"
#     if unfiltered_documents:
#         logger.info("No relevant documents — will transform query")
#         return "check_iter"
#     logger.warning("No documents available after grading — will transform query")
#     return "check_iter"


# def decide_to_generate_after_transformation(state: GraphState) -> str:
#     """Route after query transformation: retry retrieval or end."""
#     if state["question"] == "question not relevant":
#         return "query_not_at_all_relevant"
#     return "Retriever"


# def grade_generation_vs_documents_and_question(state: GraphState) -> str:
#     """Pure state reader — route based on the grade computed by grade_generation node."""
#     return state.get("generation_grade", "not useful")


# def grade_generation(state: GraphState) -> dict:
#     """Grade the generation for hallucinations and question relevance."""
#     logger.info("Checking for hallucinations")
#     question = state["question"]
#     documents = state["documents"]
#     generation = state["generation"]

#     try:
#         prompt = hallucination_prompt.invoke({"documents": documents, "generation": generation})
#         score = structured_hallucination_grader.invoke(prompt)
#         grade = score.binary_score.strip().lower()
#     except Exception:
#         logger.exception("Hallucination grading failed — treating as not useful")
#         return {"generation_grade": "not useful"}

#     if grade == "yes":
#         logger.info("Generation is grounded in documents")

#         try:
#             ans_prompt = answer_prompt.invoke({"question": question, "generation": generation})
#             score = structured_answer_grader.invoke(ans_prompt)
#             grade = score.binary_score.strip().lower()
#         except Exception:
#             logger.exception("Answer grading failed — treating as not useful")
#             return {"generation_grade": "not useful"}

#         if grade == "yes":
#             logger.info("Generation addresses the question")
#             return {"generation_grade": "useful"}
#         else:
#             logger.info("Generation does not address the question — retrying")
#             return {"generation_grade": "not useful"}
#     else:
#         logger.info("Generation is NOT grounded in documents — retrying")
#         return {"generation_grade": "not useful"}


# def route(state: GraphState) -> str:
#     """Route based on whether the max retry count has been reached."""
#     if state["count"] == state["max_count"]:
#         logger.warning("Max retry count (%d) reached", state["max_count"])
#         return "max_count_reached_end"
#     return "transform_user_query"




def retrieve_context(state: GraphState):

    

    question = state['question']

    RETRIEVER = get_or_create_retriever()

    documents = RETRIEVER.invoke(question)



    return {"documents": documents, "question": question, 'chat_history': [HumanMessage(question)]}





# --------------------------------------------- grading NODE fn--------------------------------------



def grade_documents(state: GraphState):

    print("----CHECK DOCUMENTS RELEVANCE TO THE QUESTION----")

    question = state['question']

    documents = state['documents']

    count = state['count']

    

    filtered_docs = []

    unfiltered_docs = []

    for doc in documents:

        prompt = grade_prompt.invoke({"question":question, "document":doc})

        score=structured_relevance_grader.invoke(prompt)

        grade=score.binary_score

        

        if grade=='yes':

            print("----GRADE: DOCUMENT RELEVANT----")

            filtered_docs.append(doc)

        else:

            print("----GRADE: DOCUMENT NOT RELEVANT----")

            unfiltered_docs.append(doc)

    if not filtered_docs:

        return {"unfilter_documents": unfiltered_docs,"filter_documents":[], "question": question}

    else:

        return {"filter_documents": filtered_docs,"unfilter_documents":[],"question": question}





# -------------------------------------------------- conditional to approve fn ----------------------------------



def decide_to_generate(state: GraphState):

    print("----ACCESS GRADED DOCUMENTS----")

    state["question"]

    unfiltered_documents = state["unfilter_documents"]

    filtered_documents = state["filter_documents"]

    if len(filtered_documents) > 0:

        print("----DECISION: GENERATE----")

        return "generate"

    
    if unfiltered_documents:

        print("----ALL THE DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY----")

        return "check_iter"

    

    



#---------------------------------------------------- generate answer ------------------------------------------------

def generate(state:GraphState):

    print("----GENERATE----")

    question=state["question"]

    filtered_documents=state["filter_documents"]

    
    logging.info(f"----CONTEXT DOCUMENTS: {filtered_documents}----")

    context = ""
    if len(filtered_documents)>0:
        for doc in filtered_documents:
            context += doc.page_content + "\n"
    
    

    

    generation = rag_chain.invoke({"context": filtered_documents,"question":question})

    return {"filter_documents": filtered_documents,"question":question,"generation":generation.content, 'chat_history': [AIMessage(generation.content)]}





#------------------------------------------------- transform query --------------------------------------------------

def transform_query(state:GraphState):

    question=state["question"]

    documents=state["documents"]

    

    print(f"this is my document{documents}")

    response = question_rewriter.invoke({"question":question,"documents":documents})

    print(f"----RESPONSE---- {response}")

    if response == 'question not relevant':

        print("----QUESTION IS NOT AT ALL RELEVANT----")

        return {"documents":documents,"question":response,"generation":"question was not at all relevant"}

    else:   

        return {"documents":documents,"question":response}

    

#---------------------------------------------------------- gen after transform---------------------------------------

    



def decide_to_generate_after_transformation(state:GraphState):

    question=state["question"]

    

    if question=="question not relevant":

        return "query_not_at_all_relevant"

    else:

        return "Retriever"

    



# ----------------------------------------------------------------------------------------------------------------------

def grade_generation_vs_documents_and_question(state:GraphState):

    print("---CHECK HELLUCINATIONS---")

    question= state['question']

    documents = state['documents']

    generation = state["generation"]



    prompt = hallucination_prompt.invoke({"documents":documents,"generation":generation})

    

    score = structured_hallucination_grader.invoke(prompt)

    

    grade = score.binary_score

    

    #Check hallucinations

    if grade=='yes':

        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        

        print("---GRADE GENERATION vs QUESTION ---")

        

        ans_prompt = answer_prompt.invoke({"question":question,"generation":generation})

        score = structured_answer_grader.invoke(ans_prompt)

        

        grade = score.binary_score

        

        if grade=='yes':

            print("---DECISION: GENERATION ADDRESS THE QUESTION ---")

            return "useful"

        else:

            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---TRANSFORM QUERY")

            return "not useful"

    else:

        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---TRANSFORM QUERY")

        return "not useful"





#-------------------------------------------- Check iter function-----------------------------------------

def check_iteration(state: GraphState):

    count = state['count']

    return {'count': count+1}



def route(state: GraphState):

    if state['count'] == state['max_count']:

        return 'max_count_reached_end'

    return 'transform_user_query'



def generate_fallback_answer(state: GraphState):

    print("---FALLBACK MODE: MAX RETRIES HIT. GENERATING UNVERIFIED ANSWER.---")

    

    question = state["question"]

    chat_history = state['chat_history']



    

    fallback_prompt_template = ChatPromptTemplate.from_messages(

        [

            ("system", fallback_system_prompt),

            MessagesPlaceholder(variable_name="chat_history"), # Include context

            ("human", "Final Question for General Knowledge Answer: {question}")

        ]

    )



    fallback_chain = fallback_prompt_template | model | StrOutputParser()

    

    final_answer = fallback_chain.invoke({

        "question": question,

        "chat_history": chat_history 

    })

    return {

        "generation": final_answer,

        'chat_history': [AIMessage(final_answer)]

    }


