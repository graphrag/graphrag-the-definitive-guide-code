from utils import driver, records_to_json


def register_recommendation_tool(schemas: list[dict], executors: dict):
    """Register the team recommendation tool into the agent's tool registry."""

    async def recommend_team(args):
        records, _, _ = driver.execute_query(
            """
            MATCH (c:Client {name: $clientName})-[:IN_INDUSTRY]->(ind:Industry)
            WITH c, ind
            MATCH (tech:Technology)
            WHERE tech.name IN $requiredTechnologies
            WITH c, ind, COLLECT(tech) AS techs
            UNWIND techs AS tech
            OPTIONAL MATCH (tech)<-[:SKILL_FOR]-(s:Skill)
              <-[:HAS_SKILL]-(person:Person)
            WITH c, ind, tech, person,
                 CASE WHEN person IS NOT NULL THEN 1 ELSE 0 END
                   AS hasDirectSkill
            OPTIONAL MATCH (person)-[:WORKED_ON]->(pp:Project)
              -[:FOR_CLIENT]->(cc:Client)-[:IN_INDUSTRY]->(ind)
            WITH c, person, tech,
                 hasDirectSkill,
                 CASE WHEN pp IS NOT NULL THEN 1 ELSE 0 END
                   AS industryExperience,
                 COUNT(DISTINCT pp) AS priorIndustryProjects
            WITH person,
                 SUM(hasDirectSkill) AS skillMatches,
                 SUM(industryExperience) AS industryBoost,
                 SUM(priorIndustryProjects) AS totalIndustryProjects,
                 CASE WHEN EXISTS {
                   (person)-[:WORKED_ON]->(:Project)-[:FOR_CLIENT]->(c)
                 } THEN 2 ELSE 0 END AS clientFamiliarity
            WHERE person IS NOT NULL
            RETURN person.full_name AS candidate,
                   skillMatches,
                   industryBoost,
                   clientFamiliarity,
                   totalIndustryProjects,
                   (skillMatches * 3 + industryBoost * 2 + clientFamiliarity * 4
                    + totalIndustryProjects * 0.5) AS fitScore
            ORDER BY fitScore DESC
            LIMIT 15
            """,
            clientName=args["client_name"],
            requiredTechnologies=args["required_technologies"],
        )
        return records_to_json(records)

    schemas.append(
        {
            "type": "function",
            "function": {
                "name": "recommend_team",
                "description": (
                    "Recommend the best available consultants for a new "
                    "project. Ranks candidates by a composite score based "
                    "on matching technology skills, prior experience in "
                    "the client's industry, and familiarity with the "
                    "specific client."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_name": {
                            "type": "string",
                            "description": (
                                "The exact name of the client the project "
                                "is for."
                            ),
                        },
                        "required_technologies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of technology names required for "
                                "the project (e.g. ['Kubernetes', 'AWS', "
                                "'Kafka'])."
                            ),
                        },
                    },
                    "required": ["client_name", "required_technologies"],
                },
            },
        }
    )
    executors["recommend_team"] = recommend_team
