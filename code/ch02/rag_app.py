import os
import streamlit as st
from openai import OpenAI

import chromadb

openai = OpenAI()
openai.api_key = os.environ["OPENAI_API_KEY"]
messages = [{"role": "developer", "content": """
             You are a helpful assistant to help OMG consultants to determine
             consultant placement, you have information about the consultants,
             their skills, and project experience as well as past project
             details, customers and their descriptions.
             """}]

@st.cache_resource
def load_vector_store(persist_directory="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection("documents")

def similarity_search(vectorstore, query_embedding, top_k=5):
    results = vectorstore.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

def generate_embedding(query):
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Generate a response using OpenAI LLM
def generate_response(context, query):
    prompt = f"""
    You are an assistant for OMG Consulting.
    Answer the question based on the following context.
    If the answer cannot be determined from the context, say so.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    messages.append({"role": "user",
                     "content": prompt.format(context=context, query=query)})
    response = openai.chat.completions.create(
        model="gpt-5.6",
        messages=messages,
        max_completion_tokens=200
    )
    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    return answer

# UI setup
st.set_page_config(layout="wide")
st.title("OMG Consulting Assistant")
st.write("Ask questions about our consultants, their skills, "
         "and project experience")

# Load vector store
with st.spinner("Loading knowledge base..."):
    vectorstore = load_vector_store()

# User query section
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Searching for information..."):
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Perform similarity search
        search_results = similarity_search(vectorstore, query_embedding,
                                           top_k=3)
        
        # Assemble context from search results
        context = "\n\n".join(search_results)
        
        # Generate response
        response = generate_response(context, query)
        
        # Display response
        st.write("### Answer")
        st.write(response)
        
        # Display sources (for transparency)
        st.write("### Source Documents")
        for i, doc in enumerate(search_results):
            with st.expander(f"Source {i+1}"):
                st.write(doc)
#                st.write(f"Source: {doc['metadata'].get('source', 'Unknown')}")
