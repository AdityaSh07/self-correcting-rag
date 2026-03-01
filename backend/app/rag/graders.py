from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ...prompts.prompts import (
    system_text_hallucination,
    system_text_relevance,
    system_text_answer_eval,
    system,
)
from .llm import model
from .schemas import GradeDocuments, GradeHallucinations, GradeAnswer

# --------------------------------- STRUCTURED GRADERS ------------------------

structured_relevance_grader = model.with_structured_output(GradeDocuments)
structured_hallucination_grader = model.with_structured_output(GradeHallucinations)
structured_answer_grader = model.with_structured_output(GradeAnswer)

# --------------------------------- PROMPT TEMPLATES --------------------------

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
