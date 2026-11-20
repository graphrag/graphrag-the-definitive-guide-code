import asyncio

from utils import driver, get_embedding

def create_community_index():
    driver.execute_query(
        """// tag::cypher[]
    CREATE VECTOR INDEX globalCommunity IF NOT EXISTS
    FOR (m:__Community__)
    ON m.embedding
        """  # end::cypher[]
    )

# tag::search[]
async def global_community_search(query: str) -> str:
    embedding = await get_embedding(query)

    results, _, _ = driver.execute_query(
        """MATCH (c: __Community__)
    SEARCH c IN (
        VECTOR INDEX globalCommunity
        FOR $embedding
        LIMIT 5
    ) SCORE AS score
    RETURN c.summary AS communitySummary""",
        embedding=embedding,
    )

    return "\n\n".join([el["communitySummary"] for el in results])


# end::search[]

async def main():
    create_community_index()
    output = await global_community_search("What are the main themes?")
    print(output)

if __name__ == "__main__":
    asyncio.run(main())
