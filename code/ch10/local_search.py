import asyncio

from utils import driver, get_embedding

def create_entity_index():
    driver.execute_query(
        # tag::cypher[]
        """CREATE VECTOR INDEX entitySummary IF NOT EXISTS
    FOR (m:__Entity__)
    ON m.embedding"""
        # end::cypher[]
    )

# tag::search[]
async def local_search(query: str) -> str:
    embedding = await get_embedding(query)

    records, _, _ = driver.execute_query(
        """
    MATCH (e:__Entity__)
    SEARCH e IN (
        VECTOR INDEX entitySummary
        FOR $embedding
        LIMIT 10
    ) SCORE AS score
    WITH collect(e) AS nodes
    WITH collect {
        UNWIND nodes AS n
        MATCH (n)<-[:MENTIONS]->(c:__Chunk__)
        WITH c, count(DISTINCT n) AS freq
        RETURN c.text AS chunkText
        ORDER BY freq DESC
        LIMIT 3
    } AS text_mapping, collect {
        UNWIND nodes AS n
        MATCH (n)-[:IN_COMMUNITY*]->(c:__Community__)
        WHERE c.summary IS NOT NULL
        WITH c, c.rating AS rank
        RETURN c.summary
        ORDER BY rank DESC
        LIMIT 3
    } AS report_mapping, collect {
        UNWIND nodes AS n
        MATCH (n)-[r:RELATIONSHIP_SUMMARY]-(m)
        WHERE m IN nodes
        RETURN r.summary AS descriptionText
        LIMIT 3
    } AS insideRels, collect {
        UNWIND nodes AS n
        RETURN n.summary AS descriptionText
    } AS entities
    RETURN {
        Chunks: text_mapping,
        Reports: report_mapping,
        Relationships: insideRels,
        Entities: entities
    } AS output
    """,
        embedding=embedding,
    )

    return records[0]["output"]


# end::search[]

async def main():
    create_entity_index()
    output = await local_search("What do you know about Anthropic?")
    print(output)

if __name__ == "__main__":
    asyncio.run(main())
