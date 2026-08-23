"""Gold-standard knowledge graph for the construction benchmark.

A small hand-curated reference set: one news article, the typed entities
a domain expert would extract from it, and the typed relationships
between them. The relationship types match the ontology defined in
``ontology_extractor.py``, so completeness is measured against a fully
typed gold triple set, not just unlabelled edges.
"""

# tag::kg_gold[]
GOLD_ARTICLE = (
    "Microsoft said on Friday it had completed its $69 billion acquisition "
    "of Activision Blizzard, the largest deal in gaming history. CEO Satya "
    "Nadella called it \"a turning point for our gaming business.\" The deal, "
    "first announced in January 2022, faced scrutiny from the US Federal "
    "Trade Commission and the UK's Competition and Markets Authority before "
    "receiving final approval. Activision shares closed at $95.10 on the "
    "Nasdaq."
)

# Hand-curated by a domain expert. IDs use the ALL-CAPS-HYPHENATED form the
# ontology prompt asks for; types match the ``EntityType`` enum.
GOLD_ENTITIES = [
    {"id": "MICROSOFT",                                  "type": "ORGANIZATION"},
    {"id": "ACTIVISION-BLIZZARD",                        "type": "ORGANIZATION"},
    {"id": "SATYA-NADELLA",                              "type": "PERSON"},
    {"id": "US-FEDERAL-TRADE-COMMISSION",                "type": "ORGANIZATION"},
    {"id": "UK-COMPETITION-AND-MARKETS-AUTHORITY",       "type": "ORGANIZATION"},
    {"id": "MICROSOFT-ACTIVISION-BLIZZARD-ACQUISITION",  "type": "EVENT"},
]

# Typed triples: (source, relationship_type, target). The relationship_type
# matches the ``RelationshipType`` enum and obeys the ontology's domain/range.
GOLD_RELATIONSHIPS = [
    ("MICROSOFT",                            "ACQUIRED",    "ACTIVISION-BLIZZARD"),
    ("MICROSOFT",                            "LED_BY",      "SATYA-NADELLA"),
    ("MICROSOFT",                            "INVOLVED_IN", "MICROSOFT-ACTIVISION-BLIZZARD-ACQUISITION"),
    ("ACTIVISION-BLIZZARD",                  "INVOLVED_IN", "MICROSOFT-ACTIVISION-BLIZZARD-ACQUISITION"),
    ("US-FEDERAL-TRADE-COMMISSION",          "REVIEWED",    "MICROSOFT-ACTIVISION-BLIZZARD-ACQUISITION"),
    ("UK-COMPETITION-AND-MARKETS-AUTHORITY", "REVIEWED",    "MICROSOFT-ACTIVISION-BLIZZARD-ACQUISITION"),
]
# end::kg_gold[]
