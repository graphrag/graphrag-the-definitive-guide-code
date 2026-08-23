import asyncio

from utils import driver, get_embedding

def create_relationship_index():
    driver.execute_query(
        # tag::cypher[]
        """
    CREATE VECTOR INDEX relationshipSummary IF NOT EXISTS
        FOR ()-[r:RELATIONSHIP_SUMMARY]-() ON (r.embedding)"""
        # end::cypher[]
    )

# tag::search[]
async def global_relationship_search(query: str) -> str:
    embedding = await get_embedding(query)

    results, _, _ = driver.execute_query(
        """MATCH ()-[r:RELATIONSHIP_SUMMARY]->()
    SEARCH r IN (
        VECTOR INDEX relationshipSummary
        FOR $embedding
        LIMIT 5
    ) SCORE AS score
    RETURN r.summary AS relationshipSummary""",
        embedding=embedding,
    )
    return "\n\n".join([el["relationshipSummary"] for el in results])


# end::search[]

async def main():
    create_relationship_index()
    output = await global_relationship_search("What are the main themes?")
    print(output)

if __name__ == "__main__":
    asyncio.run(main())
