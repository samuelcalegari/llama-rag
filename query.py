import os
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

LLM_MODEL = os.getenv('LLM_MODEL', 'mistral')


# Function to get the prompt templates for generating alternative questions and answering based on context
def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Tu es un assistant au sein de l'Université de Perpignan (UPVD). Ta tâche est de générer cinq 
        différentes versions de la question utilisateur donnée pour récupérer les documents pertinents à partir de la 
        base de données vectorielle. En générant plusieurs perspectives sur la question de l'utilisateur, ton 
        objectif est d'aider l'utilisateur.Fournis ces questions alternatives séparées par des nouvelles lignes.
    	Question originale: {question}""",
    )

    template = """Réponds à la question en te basant UNIQUEMENT sur le contexte suivant:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt


# Main function to handle the query process
def query(input):
    if input:
        # Initialize the language model with the specified model name
        llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
        # Get the vector database instance
        db = get_vector_db()
        # Get the prompt templates
        QUERY_PROMPT, prompt = get_prompt()

        # Set up the retriever to generate multiple queries using the language model and the query prompt
        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )

        # Define the processing chain to retrieve context, generate the answer, and parse the output
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        response = chain.invoke(input)

        return response

    return None
