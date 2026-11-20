import sys
from utils import driver

import json
import pandas as pd


def import_project_embeddings(base_path):
    embedding_df = pd.read_csv(base_path + "/project_description_embedding.csv")
    embedding_df["embedding"] = embedding_df["embedding"].apply(json.loads)

    with driver.session() as session:
        # Create person nodes and management relationships
        session.run(
            """
        UNWIND $projects AS project
        MATCH (p:Project {id: project.project_id})
        SET p.embedding = project.embedding
        """,
            {"projects": embedding_df.to_dict("records")},
        )

        print(f"Imported {len(embedding_df)} project embeddings")

        session.run("""
        CREATE VECTOR INDEX project_embedding_index IF NOT EXISTS
        FOR (p:Project) ON p.embedding
        OPTIONS { indexConfig: {
          `vector.dimensions`: 1536,
          `vector.similarity_function`: 'cosine'
        }}
        """)


def create_ft_index():
    with driver.session() as session:
        # Create person nodes and management relationships
        session.run("""
        CREATE FULLTEXT INDEX client_name_index IF NOT EXISTS
        FOR (c:Client) ON EACH [c.name]
                    """)
        print(f"Created full-text index on Client nodes")


if __name__ == "__main__":
    base_path = sys.argv[1] if len(sys.argv) > 1 else "./data/ch02/structured"
    print("Importing embedding data...")
    import_project_embeddings(base_path)
    create_ft_index()
