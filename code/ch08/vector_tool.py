from utils import driver, get_openai_embedding, records_to_json


def register_vector_tool(schemas: list[dict], executors: dict):
    """Register the vector search tool into the agent's tool registry."""

    async def vector_search_projects(args):
        query_embedding = get_openai_embedding(args["query"])

        records, _, _ = driver.execute_query(
            """
            MATCH (p:Project)
            SEARCH p IN (
              VECTOR INDEX project_embedding_index
              FOR $embedding
              LIMIT 5 * 5
            ) SCORE AS score
            RETURN p { .*, score: score } AS project
            ORDER BY score DESC
            LIMIT 5
            """,
            embedding=query_embedding,
        )
        return records_to_json(records)

    schemas.append(
        {
            "type": "function",
            "function": {
                "name": "vector_search_projects",
                "description": "Find projects semantically similar to a natural-language query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query describing the kind of project to find.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    )
    executors["vector_search_projects"] = vector_search_projects
