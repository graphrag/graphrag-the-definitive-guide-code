from utils import driver


# Drop graph if exists
driver.execute_query("CALL gds.graph.drop('entityGraph', false)")


# Project an entity graph
driver.execute_query(
    # tag::projection[]
    """
MATCH (source:__Entity__)-[r:RELATIONSHIP]-(target:__Entity__)
WHERE elementId(source) < elementId(target)
WITH DISTINCT source, target
WITH gds.graph.project(
  'entityGraph',
  source,
  target,
  {},
  {undirectedRelationshipTypes: ['*']}
) AS g
RETURN g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS rels"""
    # end::projection[]
)


driver.execute_query(
    # tag::leiden[]
    """CALL gds.leiden.write(
                     "entityGraph",
                     {writeProperty:"community",
                     includeIntermediateCommunities:True})"""
    # end::leiden[]
)


driver.execute_query(
    # tag::hierarchy[]
    """
MATCH (n:__Entity__)
WHERE n.community IS NOT NULL AND size(n.community) > 0

// 2. Connect the original node to the first level community
MERGE (c1:__Community__ {id: '0' + '-' + n.community[0]})
MERGE (n)-[:IN_COMMUNITY]->(c1)

// 3. Iterate through the community list to build the hierarchy
WITH n
UNWIND range(0, size(n.community) - 2) AS i
MERGE (child:__Community__ {id: i + '-' + n.community[i]})
SET child.level = i
MERGE (parent:__Community__ {id: i + 1 + '-' + n.community[i+1]})
SET parent.level = i + 1
MERGE (child)-[:IN_COMMUNITY]->(parent)"""
    # end::hierarchy[]
)
