from utils import driver, records_to_json


def register_fulltext_tool(schemas: list[dict], executors: dict):
    """Register the fulltext search tool into the agent's tool registry."""

    async def fulltext_search_clients(args):
        records, _, _ = driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes(
                'client_name_index', $search_term
            ) YIELD node, score
            RETURN node { .*, score: score } AS client
            ORDER BY score DESC
            """,
            search_term=args["search_term"],
        )
        return records_to_json(records)

    schemas.append(
        {
            "type": "function",
            "function": {
                "name": "fulltext_search_clients",
                "description": (
                    "Search clients by name. Supports fuzzy ('Acme~'), "
                    "wildcard ('Acm*'), and boolean ('Acme AND Corp') syntax."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {
                            "type": "string",
                            "description": (
                                "The search term to match "
                                "against client names."
                            ),
                        },
                    },
                    "required": ["search_term"],
                },
            },
        }
    )
    executors["fulltext_search_clients"] = fulltext_search_clients
