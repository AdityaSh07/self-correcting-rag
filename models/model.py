from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from prompts.prompts import check_answer, system_text

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



#-------------------------------------- CREATE STATE --------------------------------
class GraphState(TypedDict):
    
    chat_history: Annotated[list[BaseMessage], add_messages]
    answer: AIMessage
    retrieved_context: str
    evaluation: Literal['REVISE',"ACCEPT"]
    iteration: int

# to be provided
    max_iteration: int
    user_query: HumanMessage  



# ----------------------------------- retrieve_context NODE fn-----------------------------------
def retrieve_context(state: GraphState):
    
    user_query = state['user_query'].content
    
    # Context chain adapted from your original structure
    context_chain = (
        RETRIEVER 
        | RunnableLambda(format_docs) 
        | StrOutputParser()
    )
    
    retrieved_context = context_chain.invoke(user_query) # human msg to str 

    return {'retrieved_context': retrieved_context, 'chat_history': [HumanMessage(user_query)]}


# --------------------------------------------- gen_result NODE fn--------------------------------------

def generate_result(state: GraphState):
    user_query = state['user_query'].content
    retrieved_context = state['retrieved_context']
    chat_history = state['chat_history']

    # Templating for the initial answer generation
    template = ChatPromptTemplate(
        [
            ('system', system_text.format(retrieved_context="{retrieved_context}")),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{user_query}')
        ]
    )
    prompt = template.invoke({
        'retrieved_context': retrieved_context, 
        'chat_history': chat_history, 
        'user_query': user_query
    })
    
    return {'answer': model.invoke(prompt).content}


# -------------------------------------------------- conditional to approve fn ----------------------------------

def evaluate(state: GraphState):
    user_query = state['user_query'].content
    retrieved_context = state['retrieved_context']
    answer = state['answer']
    iteration = state['iteration']+1

    answer_evaluate = check_answer.format(user_query = user_query, retrieved_context  = retrieved_context, answer = answer)

    evaluation = model.invoke(answer_evaluate).content

    if iteration > state['max_iteration']:
        evaluation = 'ACCEPT'

    if evaluation == 'ACCEPT':
        return {
            'evaluation': evaluation,
            'chat_history': [AIMessage(answer)],  # add_messages will handle appending
            'iteration': iteration
        }

    return {
        'evaluation': evaluation,
        'iteration': iteration
    }


def route_evaluation(state: GraphState):

    if state['evaluation'] == 'ACCEPT':
        return 'approved'
    else:
        return 'needs_improvement'


     

#-------------------------------------------- graph creation --------------------------------------------------------------

graph = StateGraph(GraphState)

graph.add_node('retrieve_context', retrieve_context)
graph.add_node('generate_result', generate_result)
graph.add_node('evaluate', evaluate)

graph.add_edge(START, 'retrieve_context')
graph.add_edge('retrieve_context', 'generate_result')
graph.add_edge('generate_result', 'evaluate')
graph.add_conditional_edges('evaluate',route_evaluation, {'approved': END, 'needs_improvement':'generate_result'})


#-------------------------- CREATE CHECKPOINTER AND WORKFLOW ------------------------------------
checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)










     



    








     
     














