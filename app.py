import os
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import google.generativeai as genai

# Import LlamaIndex modules
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# -----------------------------------------------------------------------------
# Configure Gemini API with your Google API key
load_dotenv() 
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found in environment variables. Please check your .env file.")
else:
    # Configure Gemini API with your API key
    genai.configure(api_key=api_key)

# -----------------------------------------------------------------------------
# Streamlit UI Layout
st.title("📖 Q&A System with Gemini & LlamaIndex")
st.write("Upload your documents (TXT, PDF) to build the Q&A system.")

# File uploader for TXT and PDF files (PDF text extraction can be enhanced as needed)
uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info("Saving uploaded files and building document index...")
    
    # Create a temporary directory for uploaded documents
    docs_path = Path("uploaded_docs")
    docs_path.mkdir(exist_ok=True)
    
    # Save each uploaded file to the directory
    for uploaded_file in uploaded_files:
        file_path = docs_path / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.success("Files uploaded successfully!")
    
    # -----------------------------------------------------------------------------
    # Load documents using SimpleDirectoryReader
    documents = SimpleDirectoryReader(str(docs_path)).load_data()
    
    # -----------------------------------------------------------------------------
    # Create a node parser using defaults (no explicit chunk size or overlap settings)
    parser = SentenceWindowNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    
    # -----------------------------------------------------------------------------
    # Configure Gemini Embeddings and LLM using the Settings approach
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
    Settings.llm = Gemini(models="gemini-pro")
    
    # Build the vector store index from the nodes
    index = VectorStoreIndex(nodes)
    
    # Create a query engine that retrieves the top 3 most similar chunks
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    st.success("Indexing complete! You can now ask questions based on your documents.")
    
    # -----------------------------------------------------------------------------
    # Accept user queries and display answers
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Generating answer..."):
            response = query_engine.query(query)
        st.subheader("📌 Answer:")
        st.write(response.response)
