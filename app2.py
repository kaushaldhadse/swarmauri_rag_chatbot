from swarmauri.llms.concrete.GroqModel import GroqModel
from swarmauri.vector_stores.concrete.TfidfVectorStore import TfidfVectorStore
from swarmauri.documents.concrete.Document import Document
from swarmauri.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from swarmauri.messages.concrete.SystemMessage import SystemMessage
from swarmauri.messages.concrete.HumanMessage import HumanMessage
from swarmauri.agents.concrete.RagAgent import RagAgent

import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv('Chatgroq_API')

# Initialize the LLM
def initialize_llm():
    if API_KEY:
        llm = GroqModel(api_key=API_KEY)
        st.success("LLM Initialized Successfully")
        return llm
    else:
        st.error("API Key not found")
        return None

# Initialize the vector store and add documents
def initialize_vector_store():
    vector_store = TfidfVectorStore()

    folder_path = "C:\\Python_Programs\\Swarmauri\\documents"

    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                content = file.read()
            
            documents.append(Document(content=content))
            

    vector_store.add_documents(documents)
    st.info("Documents added to vector store successfully")
    return vector_store

# Get allowed models
def get_allowed_models(llm):
    failing_llms = [
        "llama3-70b-8192",
        "llama3.2-90b-text-preview",
        "mixtral-8x7b-32768",
        "llava-v1.5-7b-4096-preview",
        "llama-guard-3-8b",
    ]
    return [model for model in llm.allowed_models if model not in failing_llms]

# Initialize the RAG Agent
def initialize_rag_agent(llm, vector_store):
    allowed_models = get_allowed_models(llm)
    if allowed_models:
        llm.name = allowed_models[1]  # Use the second allowed model
    else:
        st.error("No allowed models found")
        return None

    rag_system_context = "You are a helpful assistant that provides answers to the user. Only use the details below:"

    rag_conversation = MaxSystemContextConversation(
        system_context=SystemMessage(content=rag_system_context), max_size=8
    )

    rag_agent = RagAgent(
        llm=llm,
        conversation=rag_conversation,
        system_context=rag_system_context,
        vector_store=vector_store,
    )

    st.success("RAG Agent Initialized Successfully")
    return rag_agent

# Function to process a prompt and display the response
def process_prompt(prompt, rag_agent):
    response = rag_agent.exec(prompt)
    return response

# Streamlit App
def main():
    st.title("Swarmauri RAG Agent Language Comparison")

    # Initialize LLM, Vector Store, and RAG Agent
    llm = initialize_llm()
    if not llm:
        return

    vector_store = initialize_vector_store()
    rag_agent = initialize_rag_agent(llm, vector_store)
    if not rag_agent:
        return

    # Input prompt from the user
    prompt = st.text_input("Enter your prompt:")
    if st.button("Submit"):
        if prompt.strip():
            response = process_prompt(prompt, rag_agent)
            st.write(f"Response: {response}")
        else:
            st.warning("Please enter a valid prompt.")

if __name__ == "__main__":
    main()
