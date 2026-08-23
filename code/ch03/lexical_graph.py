import sys
import os
import glob
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
import hashlib

load_dotenv("graphrag-book.env")

# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

# tag::setup_schema[]
# Set up the database schema
def setup_schema():
    with driver.session() as session:
        session.run("""
        CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE
        """)
        session.run("""
        CREATE CONSTRAINT document_fileName IF NOT EXISTS FOR (d:Document) REQUIRE d.fileName IS UNIQUE
        """)
        # Create vector index for embeddings
        session.run("""
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS { indexConfig: { `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
          } } """)
        # Enforce FLOAT32 precision and 1536 dimensions on the embedding property
        session.run("""
        CREATE CONSTRAINT chunk_embedding_type IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.embedding IS :: VECTOR<FLOAT32>(1536)
        """)
# end::setup_schema[]

# tag::process_documents[]
# Load and process documents
def process_documents(base_path="./data"):
    documents = []
    files = glob.glob(os.path.join(base_path, f"**/*.md"), recursive=True)
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({"content": content, "source": file})
    return documents

# Split documents into chunks
# TODO use semantic chunks
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    chunks = []
    for doc in documents:
        text = doc["content"]
        source = doc["source"]
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_text = text[i:i + chunk_size]
            chunks.append({"id": hashlib.md5(chunk_text.encode("utf-8")).hexdigest(), "text": chunk_text, "source": source})
    return chunks

# Generate embeddings using OpenAI
def generate_embeddings(texts):
    response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    return [embedding.embedding for embedding in response.data]
# end::process_documents[]

# tag::store_lexical_graph[]
# Store chunks in Neo4j
def store_lexical_graph(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings_list = generate_embeddings(texts)
    source = None
    position = 0
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            if source != chunk["source"]:
                source = chunk["source"]
                position = 0
            else:
                position += 1
            session.run("""
            MERGE (d:Document {fileName: $source})
            MERGE (c:Chunk { id: $id })
            ON CREATE SET c.text=$text, c.position = $position
            WITH * 
            SET c.embedding = vector($embedding, 1536, FLOAT32)
            MERGE (c)-[:PART_OF]->(d)
            """, {
                "source": source,
                "id": chunk["id"],
                "position": position,
                "text": chunk["text"],
                "embedding": embeddings_list[i]
            })
# end::store_lexical_graph[]
#           
# Main execution
if __name__ == "__main__":
    base_path = sys.argv[1] if len(sys.argv) > 1 else "./data"

    print("Setting up Neo4j schema...")
    setup_schema()
    
    print("Processing documents...")
    documents = process_documents(base_path)
    print(f"Loaded {len(documents)} documents")    
    print("Splitting documents into chunks...")

    chunks = split_documents(documents)
    store_lexical_graph(chunks)
    
    print("Lexical graph construction complete!")
    driver.close()