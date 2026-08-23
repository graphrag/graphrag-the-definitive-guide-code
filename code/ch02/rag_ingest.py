import os
import glob
from openai import OpenAI
import chromadb
import sys

openai = OpenAI()

# tag::load_documents[]
def load_documents(base_path):
    documents = []
    
    # Load resumes from cv/*.md
    resume_path = os.path.join(base_path, "cv", "*.md")
    for file in glob.glob(resume_path):
        try:
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"content": f.read(), "source": file})
                print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Load project descriptions from projects/prj_*.md
    projects_path = os.path.join(base_path, "projects", "prj_*.md")
    for file in glob.glob(projects_path):
        try:
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"content": f.read(), "source": file})
                print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return documents
# end::load_documents[]

# tag::process_documents[]
# Text splitting - creating semantically meaningful chunks
# In the real world, you would use a more sophisticated text splitter, 
# like splitting on headers or paragraphs
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    chunks = []
    for doc in documents:
        text = doc["content"]
        source = doc["source"]
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            chunks.append({"content": chunk, "source": source})
    return chunks

# Create embeddings and store in Chroma
def create_vector_store(chunks, persist_directory="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection("documents")
    
    for i, chunk in enumerate(chunks):
        response = openai.embeddings.create(
            input=chunk["content"],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        collection.add(
            documents=[chunk["content"]],
            metadatas=[{"source": chunk["source"]}],
            ids=[f"chunk_{i}"],
            embeddings=[embedding]
        )
    return collection
# end::process_documents[]

if __name__ == "__main__":
    base_path = sys.argv[1] if len(sys.argv) > 1 else "./data"
    
    print("Loading documents...")
    documents = load_documents(base_path)
    print(f"Loaded {len(documents)} documents")
    
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating vector embeddings and storing in Chroma...")
    vectorstore = create_vector_store(chunks)
    print("Ingestion complete!")