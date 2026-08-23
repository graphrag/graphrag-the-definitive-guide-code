# app.py (Streamlit application with GraphRAG)
import os
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from graphrag_retrievers import GraphRAGApp

# Load environment variables
load_dotenv("graphrag-book.env")

# Neo4j connection (with error handling)
@st.cache_resource
def init_neo4j():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        return driver, None
    except Exception as e:
        return None, str(e)

# Initialize Neo4j and GraphRAG
driver, connection_error = init_neo4j()
if driver:
    graphrag = GraphRAGApp(driver)

# Application UI
st.title("OMG Consulting Assistant")
st.write("Ask questions about our consultants, their skills, and project experience")

# Display connection error if any
if connection_error:
    st.error(f"Failed to connect to Neo4j: {connection_error}")
    st.stop()

# Retrieval method selection
retrieval_method = st.selectbox(
    "Retrieval Method",
    ["auto", "vector", "entity", "relationship", "hybrid"],
    index=0,
    help="Select how the assistant should search for information"
)

# Query input
query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching for information..."):
        # Execute query
        result = graphrag.answer_query(query, retrieval_method)
        
        # Display response
        st.write("### Answer")
        st.write(result["answer"])
        
        # Display source documents
        with st.expander("View Source Information"):
            for i, doc in enumerate(result["documents"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc["content"])
                
                # Display metadata for transparency
                st.markdown("**Metadata:**")
                metadata_str = "\n".join([f"- {k}: {v}" for k, v in doc["metadata"].items() 
                                         if k not in ["content", "embedding"]])
                st.text(metadata_str)
                st.markdown("---")
        
        # Display retrieval method used
        st.info(f"Retrieval Method: {result['retrieval_method']}")

# Session cleanup
if st.session_state.get("_is_running") is False:
    # This will run when the app is shutting down
    if driver:
        driver.close()