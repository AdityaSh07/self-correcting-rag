from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from operator import add

from pydantic import BaseModel, Field

from prompts.prompts import system_text_hallucination, system_text_relevance, system_text_answer_eval, system, fallback_system_prompt

from langchain import hub

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langgraph.graph import StateGraph, START, END

from typing import TypedDict, Annotated,Literal

from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import InMemorySaver



load_dotenv()



# MODELS DEFINED

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature=0.8)

embed_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

rag_prompt = hub.pull("rlm/rag-prompt")

rag_chain = rag_prompt | model





# --------------------------------- CREATE RETRIEVER ---------------------------

def get_retriever():

         # LOADER

    loader = DirectoryLoader(path='documents',

                            glob='*.txt',

                            loader_cls=TextLoader)



    documents = loader.load()





    # TEXT SPLIT

    text_splitter = RecursiveCharacterTextSplitter(

            chunk_size = 500,

            chunk_overlap = 50

        )



    chunks = text_splitter.split_documents(documents)





    # VECTOR STORE

    vec_store = Chroma(

        embedding_function= embed_model,

        persist_directory='chroma_db',

        collection_name='embeddings'

    )

    vec_store.add_documents(chunks)





    # SETUP RETRIEVER AND CONTEXT

    retriever = vec_store.as_retriever(

        search_type = 'mmr',

        search_kwargs={'k': 5, 'lambda_mult': 0.8})



    return retriever



RETRIEVER = get_retriever()



def format_docs(retrieved_docs):

    """Formats retrieved documents into a single string."""

    return '\n\n'.join(docs.page_content for docs in retrieved_docs)





# ------------------- GRADING -----------------------------------------------





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





#--------------------------------------Write ques again -----------------------------

     

re_write_prompt = ChatPromptTemplate.from_messages(

    [

        ("system", system),

        (

            "human","""Here is the initial question: \n\n {question} \n,

             Here is the document: \n\n {documents} \n ,

             Formulate an improved question. if possible other return 'question not relevant'."""

        ),

    ]

)

question_rewriter = re_write_prompt | model | StrOutputParser()



#-------------------------------------- CREATE STATE --------------------------------

class GraphState(TypedDict):

    

    chat_history: Annotated[list[BaseMessage], add_messages]

    question: str

    generation: str

    documents: list[str]

    filter_documents: list[str]

    unfilter_documents: list[str]

    count: int

    max_count: int







# ----------------------------------- retrieve_context NODE fn-----------------------------------

def retrieve_context(state: GraphState):

    

    question = state['question']



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

    if len(unfiltered_docs)>1:

        return {"unfilter_documents": unfiltered_docs,"filter_documents":[], "question": question}

    else:

        return {"filter_documents": filtered_docs,"unfilter_documents":[],"question": question}





# -------------------------------------------------- conditional to approve fn ----------------------------------



def decide_to_generate(state: GraphState):

    print("----ACCESS GRADED DOCUMENTS----")

    state["question"]

    unfiltered_documents = state["unfilter_documents"]

    filtered_documents = state["filter_documents"]

    

    

    if unfiltered_documents:

        print("----ALL THE DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY----")

        return "check_iter"

    if filtered_documents:

        print("----DECISION: GENERATE----")

        return "generate"

    



#---------------------------------------------------- generate answer ------------------------------------------------

def generate(state:GraphState):

    print("----GENERATE----")

    question=state["question"]

    documents=state["documents"]

    

    generation = rag_chain.invoke({"context": documents,"question":question})

    return {"documents":documents,"question":question,"generation":generation.content, 'chat_history': [AIMessage(generation.content)]}





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

        "not useful"





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



# -------------------------------------------- graph creation --------------------------------------------------------------



graph = StateGraph(GraphState)

graph.add_node("vector_retrieved_docs", retrieve_context)

graph.add_node("grading_documents", grade_documents) 

graph.add_node("content_generator", generate)

graph.add_node("transform_user_query", transform_query)

graph.add_node('check_iteration', check_iteration)

graph.add_node("generate_fallback_answer", generate_fallback_answer)





# --- EDGES ---



# 1. Start -> Retrieve

graph.add_edge(START, "vector_retrieved_docs")



# 2. Retrieve -> Grade Docs

graph.add_edge("vector_retrieved_docs", "grading_documents")



# 3. Grade Docs -> Decide Path 

graph.add_conditional_edges("grading_documents",

                            decide_to_generate,

                            {

                            "generate": "content_generator",

                            "check_iter": "check_iteration" 

                            })



# 4. Content Generator -> Decide Path

graph.add_conditional_edges("content_generator",

                            grade_generation_vs_documents_and_question,

                            {

                            "useful": END,

                            "not useful": "check_iteration",

                            })



# 5. Check Iteration -> Decide Loop Fate

graph.add_conditional_edges("check_iteration",

                            route,

                            {

                            "transform_user_query": "transform_user_query",

                            "max_count_reached_end": "generate_fallback_answer" 

                            })



graph.add_conditional_edges("transform_user_query",

                            decide_to_generate_after_transformation,

                            {

                            "Retriever": "vector_retrieved_docs",

                            "query_not_at_all_relevant": END 

                            })

# 7. Final Apology Node -> End

graph.add_edge("generate_fallback_answer", END)


#-------------------------- CREATE CHECKPOINTER AND WORKFLOW ------------------------------------

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)









     



    








     
     














