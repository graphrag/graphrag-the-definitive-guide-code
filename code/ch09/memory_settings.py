import os

from dotenv import load_dotenv
from pydantic import SecretStr

from neo4j_agent_memory import MemorySettings, Neo4jConfig
from neo4j_agent_memory.config.settings import (
    SchemaConfig,
    SchemaModel,
    ExtractionConfig,
    ExtractorType,
    EmbeddingConfig,
)

load_dotenv("graphrag-book.env")


# OMG's six existing entity types (see chapter 3). These replace the library's
# default POLE+O ontology for the long-term region.
OMG_ENTITY_TYPES = [
    "Person",
    "Project",
    "Client",
    "Skill",
    "Technology",
    "Industry",
]


def build_memory_settings() -> MemorySettings:
    """Build a MemorySettings configured against OMG's knowledge graph.

    The custom ontology overrides POLE+O so the library's MENTIONS edges
    land on existing OMG nodes (after `adopt_omg_graph.py` has been run).
    The three-stage extraction pipeline (spaCy fast, GLiNER2 zero-shot,
    LLM fallback) handles the cost/quality tradeoff: most messages settle
    at spaCy or GLiNER2; the LLM only runs when the cheaper extractors
    fail to find domain entities.
    """
    return MemorySettings(
        neo4j=Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=SecretStr(os.getenv("NEO4J_PASSWORD", "password")),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        ),
        schema_config=SchemaConfig(
            model=SchemaModel.CUSTOM,
            entity_types=OMG_ENTITY_TYPES,
            strict_types=True,
            enable_subtypes=False,
        ),
        extraction=ExtractionConfig(
            extractor_type=ExtractorType.PIPELINE,  # spaCy -> GLiNER2 -> LLM
            entity_types=OMG_ENTITY_TYPES,
            extract_preferences=True,
            gliner_model="urchade/gliner_medium-v2.1",
            spacy_model="en_core_web_sm",
            llm_fallback_model=os.getenv("OPENAI_MODEL", "gpt-5"),
        ),
        embedding=EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )
