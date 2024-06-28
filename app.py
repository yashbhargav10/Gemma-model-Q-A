import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Filepath to save chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Function to serialize Document objects to a JSON-compatible format
def serialize_document(doc):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

# Function to deserialize Document objects from a JSON-compatible format
def deserialize_document(doc_dict):
    return Document(page_content=doc_dict["page_content"], metadata=doc_dict["metadata"])

# Function to save chat history to a JSON file
def save_chat_history():
    serializable_history = []
    for query, answer, response_time, context in st.session_state.chat_history:
        serialized_context = [serialize_document(doc) for doc in context]
        serializable_history.append((query, answer, response_time, serialized_context))
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(serializable_history, f)

# Function to load chat history from a JSON file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            serialized_history = json.load(f)
            st.session_state.chat_history = [
                (query, answer, response_time, [deserialize_document(doc) for doc in context])
                for query, answer, response_time, context in serialized_history
            ]
    else:
        st.session_state.chat_history = []

# Initialize chat history
if "chat_history" not in st.session_state:
    load_chat_history()

# Streamlit app title
st.title("Gemma Model Document Q&A Chatbot")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

# Function to create and load vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader(directory_path="./GM Manuals")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Sidebar for creating document embeddings
st.sidebar.button("Create Document Embeddings", on_click=vector_embedding)
if "vectors" in st.session_state:
    st.sidebar.success("Vector Store DB is ready.")
else:
    st.sidebar.warning("Click the button to create the Vector Store DB.")

# Function to process user query
def process_query(query):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': query})
    response_time = time.process_time() - start
    
    st.session_state.chat_history.append((query, response['answer'], response_time, response["context"]))
    save_chat_history()  # Save chat history after adding a new entry

# Chatbot UI
st.subheader("Chat with Gemma Model")

# User input form
with st.form(key='query_form', clear_on_submit=True):
    user_query = st.text_input("Enter your question:", "")
    submit_button = st.form_submit_button(label='Send')

# Process the query and update chat history
if submit_button and user_query:
    process_query(user_query)

# Display chat history in a WhatsApp-like interface
for query, answer, response_time, context in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        st.write(answer)

st.sidebar.markdown("## Chat History")
for query, answer, response_time, context in st.session_state.chat_history:
    st.sidebar.write(f"**Q:** {query}")
    st.sidebar.write(f"**A:** {answer}")

