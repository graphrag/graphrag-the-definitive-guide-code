"""Evaluation questions for the Chapter 8 agent.

Each question is annotated with a reference answer and with the tools the
agent *should* use. The eval scores the agent on four axes:

    task_completion       - did the agent produce any answer at all?
    correctness           - does the answer align with the reference?
    tool_correctness      - were the called tools appropriate for the task?
    trajectory_match      - did the expected tools appear in the trace?

Questions are chosen to cover every tool the agent has (schema-read, Cypher,
vector similarity, fulltext, recommendation) and three difficulty tiers:
single-tool lookups, multi-hop reasoning, and out-of-scope refusals.
"""

EVAL_QUESTIONS = [
    # tag::questions[]
    {
        "id": "project_semantic_search",
        "difficulty": "simple",
        "question": "Find projects related to cloud migration strategy.",
        "reference": (
            "The 'Cloud Migration Strategy' project is the most directly relevant: "
            "an enterprise-wide multi-cloud migration project."
        ),
        "expected_tools_any": ["vector_search_projects"],
        # Vector search doesn't need the schema; flag it as wasted work.
        "forbidden_tools": ["get-schema"],
    },
    {
        "id": "multi_hop_healthcare",
        "difficulty": "hard",
        "question": (
            "Who worked on projects in the healthcare industry and also has data "
            "governance or machine learning skills?"
        ),
        "reference": (
            "Sarah Johnson is the only matching consultant. She has both Data "
            "Governance and Machine Learning skills, and worked on four "
            "healthcare-industry projects: Healthcare Data Modernization, "
            "Healthcare AI Implementation, Healthcare GraphRAG Implementation, "
            "and Healthcare Knowledge Management."
        ),
        "expected_tools_any": ["read-cypher"],
    },
    # ... four more questions spanning every tool and every difficulty tier
    # end::questions[]
    {
        "id": "simple_skill_lookup",
        "difficulty": "simple",
        "question": "What skills does Jennifer Rodriguez have?",
        "reference": (
            "Jennifer Rodriguez has skills in Microservices Architecture, API-First "
            "Design, Cloud-Native Applications, GraphRAG, Event-Driven Architecture, "
            "Domain-Driven Design, and Integration Patterns."
        ),
        "expected_tools_any": ["read-cypher"],
        "forbidden_tools": [],
    },
    {
        "id": "client_projects_lookup",
        "difficulty": "simple",
        "question": "Which consultants worked on projects for First National Bank?",
        "reference": (
            "Sarah Chen, Michelle Wong, Daniel Singh, Benjamin Okonkwo, and Carlos "
            "Rodriguez all worked on the Digital Banking Platform project for First "
            "National Bank."
        ),
        "expected_tools_any": ["read-cypher", "fulltext_search_clients"],
        "forbidden_tools": [],
    },
    {
        "id": "team_recommendation",
        "difficulty": "medium",
        "question": (
            "Recommend a small team for a new project at First National Bank that "
            "will need Kubernetes and Cloud Security skills."
        ),
        "reference": (
            "A good team would combine consultants with direct Kubernetes and Cloud "
            "Security skills, prior Financial Services industry experience, and ideally "
            "familiarity with First National Bank. The `recommend_team` tool scores "
            "candidates exactly on those three dimensions."
        ),
        "expected_tools_any": ["recommend_team"],
        # The recommender doesn't need the schema either.
        "forbidden_tools": ["get-schema"],
    },
    {
        "id": "out_of_scope_refusal",
        "difficulty": "refusal",
        "question": "What is the home address of Jennifer Rodriguez?",
        "reference": (
            "The graph only stores office location, email, and phone for a Person, "
            "not a home address. The agent should state that the information is not "
            "available rather than invent one."
        ),
        "expected_tools_any": ["get-schema", "read-cypher"],
        "forbidden_tools": [],
    },
]
