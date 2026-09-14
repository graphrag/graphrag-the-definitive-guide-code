# graphrag_retrievers.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Helper: get OpenAI embedding for a string
def get_openai_embedding(text: str, model: str = "text-embedding-small") -> list:
    response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding

# Helper: call OpenAI completion
# Generate a response using OpenAI LLM
def generate_response(context, query):
    system_prompt = f"""
    You are an helpful assistant for OMG Consulting to help with questions about projects, staff, clients 
    based on data from their internal knowledge source
    Answer the question only based on the following context.
    If the answer cannot be determined from the context, say so.
    """

    prompt = f"""
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt}]

    response = openai.chat.completions.create(
        model="gpt-5.6",
        messages=messages,
        max_completion_tokens=200
    )
    answer = response.choices[0].message.content
    return answer

class GraphRAGRetriever:
    def __init__(self, driver):
        self.driver = driver

# tag::vector_retriever[]
    def vector_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Basic vector similarity retrieval using Neo4j vector index
        """
        query_embedding = get_openai_embedding(query)
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c:Chunk)
            SEARCH c IN (
              VECTOR INDEX chunk_embedding
              FOR $embedding
              LIMIT $top_k * 5
            ) SCORE AS score
            RETURN c.id AS id, c.text AS text, c.source AS source, score
            ORDER BY score DESC
            LIMIT $top_k
            """, {"embedding": query_embedding, "top_k": top_k})
            docs = []
            for record in result:
                docs.append({
                    "content": record["text"],
                    "metadata": {
                        "id": record["id"],
                        "source": record["source"],
                        "score": record["score"]
                    }
                })
            return docs
# end::vector_retriever[]

# tag::vector_entity_retriever[]
    def vector_chunk_entity_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Vector search for chunks, then expand to directly related entities and related entities
        """
        query_embedding = get_openai_embedding(query)
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c:Chunk)
            SEARCH c IN (
              VECTOR INDEX chunk_embedding
              FOR $embedding
              LIMIT $top_k * 5
            ) SCORE AS score
            WITH c, score
            ORDER BY score DESC
            LIMIT $top_k
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
            OPTIONAL MATCH (e)-[rel]-(e2:!Chunk)
            WITH c, score, collect({
                source:e.id,
                sourceType:[l IN labels(e) WHERE l <> '__Entity__'][0],
                rel:type(rel),
                target:e2.id,
                targetType:[l IN labels(e2) WHERE l <> '__Entity__'][0]}) AS entities
            RETURN c.id AS id, c.text AS text, c.source AS source, score, entities
            """, {"embedding": query_embedding, "top_k": top_k})
            docs = []
            for record in result:
                entities_str = ""
                if record["entities"]:
                    entities_str += "\nEntities: " + ", ".join([
                        f"({e['sourceType']}:{e['source']}) ({e['rel']}) ({e['targetType']}:{e['target']})"
                        for e in record["entities"] if e
                    ])
                docs.append({
                    "content": record["text"] + entities_str,
                    "metadata": {
                        "id": record["id"],
                        "source": record["source"],
                        "score": record["score"],
                        "entities": record["entities"],
                    }
                })
            return docs
# end::vector_entity_retriever[]

class GraphRAGApp:
    def __init__(self, driver):
        self.retriever = GraphRAGRetriever(driver)

# tag::answer_question[]
    def answer_question(self, question: str, retrieval_method: str = "vector") -> Dict[str, Any]:
        """
        Answer a query using the specified retrieval method and OpenAI LLM

        retrieval_method options:
        - "vector": Basic vector similarity search
        - "vector_entity": Vector search, then expand to entities and related entities
        """
        if retrieval_method == "vector":
            docs = self.retriever.vector_retrieval(question)
        elif retrieval_method == "vector_entity":
            docs = self.retriever.vector_chunk_entity_retrieval(question)
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")

        context = []
        for i, doc in enumerate(docs):
            context.append(f"""
            Document {i+1}:
            {doc['content']}
            {doc['metadata']}
            """)

        context_string = "\n\n".join(context)
        response = generate_response(context_string, question)
        return {
            "question": question,
            "answer": response,
            "documents": docs,
            "retrieval_method": retrieval_method
        }
# end::answer_question[]

# Example usage
if __name__ == "__main__":
    app = GraphRAGApp(driver)
    test_queries = [
        "What skills does Emily Johnson have?",
        "Which consultants have experience with healthcare clients?",
        "What projects has Emily Johnson worked on that involved AWS?",
        "Who in our company knows both Neo4j and React?",
        "Which clients have we done cloud migration projects for?"
    ]
    for question in test_queries:
        print("\n" + "="*80)
        print(f"Question: {question}\n")
        result_vector = app.answer_question(question, retrieval_method="vector")
        print(f"Answer Vector: {result_vector['answer']}\n")
        result_graphrag = app.answer_question(question, retrieval_method="vector_entity")
        print(f"Answer GraphRAG:\n{result_graphrag['answer']}\n\n")
#        print("\nRetrieved Documents:")
#        for i, doc in enumerate(result_graphrag['documents']):
#            print(f"\nDocument {i+1}:")
#            print(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])
    driver.close()