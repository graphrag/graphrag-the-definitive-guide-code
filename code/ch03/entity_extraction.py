# entity_extraction.py
import os
import openai
from openai import OpenAI
import json
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel
import typing

# Load environment variables
load_dotenv("graphrag-book.env")

# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

client = OpenAI()

# tag::prompts[]
SYSTEM_PROMPT = """
Your job is to extract entities and relevant relationships from the provided
text and return them in a structured JSON format.

The entities and relationships must follow the provided graph schema, matching
the types and attributes defined in the schema.
Making up new types and attributes is NOT allowed.

{schema}

During the extraction you must always consult this list of existing entities,
to determine if an entity already exists in the system and can be mapped to an
already existing value.
In that case do NEVER make up a new entity for that existing value and also do
not add other attributes to the existing entities, just re-use the existing
Type and ID.
When extracting new entities use a consistent ID and Type that matches the
format provided for the existing entities.

Here is the list of existing entities:
{existing_entities}

The required structured JSON output format is:

```json
entities:[
    {{type:"Type", id:"consistent-id", attributes: {{key: "value", key2: value2}}}}
],
relationships:[
    {{type:"TYPE", sourceId:"source-id", sourceType:"SourceType",
      targetId:"target-id", targetType:"TargetType",
      attributes: {{key: "value", key2: value2}}}}
]
```

Here is an example for a text and a structured output derived from that text.

```
{{type: "Person", id: <id>, full_name: <full_name>,
  [role: <role>, department: <department>, email: <email>,
   joined: <joined_date>]}}
```

Return only valid JSON with the extracted entities and relationships, nothing
else, no explanations, apologies, additions.
If there are no entities or relationships to be found in the text, just return
an empty JSON object `{{}}`.
"""

EXTRACTION_PROMPT = """
Extract entities and relationships from the given text, optional attributes can be added to the entities and relationships. 
Optional attributes indicated in square brackets `{{name:String, [...]}}` in the schema attribute definitions.

{text}
"""

SCHEMA = """
// Nodes:
(:Person {id: String, full_name: String,
    [role: String, department: String, email: String, phone: String,
     office_location: String, expertise: String, seniority: String,
     joined: Date]})
(:Skill {name: String})
(:Project {id: String, name: String, description: String, start_date: Date,
    [end_date: Date, status: String, project_type: String,
     budget_range: String, location: String]})
(:Client {id: String, name: String})
(:Industry {name: String})
(:Technology {name: String, category: String})

// Relationships:
(:Person)-[:MANAGES]->(:Person)
(:Person)-[:HAS_SKILL {proficiency: String,
    [category: String, proficiency_level: String, certification: String,
     skill_description: String, project_acquired: String,
     acquisition_date: Date, last_used: Date, description: String]}]->(:Skill)
(:Project)-[:FOR_CLIENT]->(:Client)
(:Client)-[:IN_INDUSTRY]->(:Industry)
(:Person)-[:WORKED_ON {role: String,
    [start_date: Date, end_date: Date, key_contributions: String]}]->(:Project)
(:Project)-[:USED_TECHNOLOGY {context: String}]->(:Technology)
(:Skill)-[:SKILL_FOR]->(:Technology)
"""
# end::prompts[]

# tag::response[]
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional, Union, List

class FlexibleAttribute(BaseModel):
    key: str
    value: str

#class FlexibleAttributes(BaseModel):
#    model_config = ConfigDict(extra="allow", json_schema_extra={"additionalProperties": True})

class Entity(BaseModel):
    type: str
    id: str
    attributes: Optional[Dict[str, Union[str, float, bool]]] = {}

class Relationship(BaseModel):
    type: str
    sourceId: str
    sourceType: str
    targetId: str
    targetType: str
    attributes: Optional[Dict[str, Union[str, float, bool]]] = {}

class ExtractionResponse(BaseModel):
    entities: list[Entity] = []
    relationships: list[Relationship] = []
# end::response[]

# tag::extraction[]
def extract_entities(chunk_text, schema, existing_entities):
    system_prompt = SYSTEM_PROMPT.format(
        schema=schema,
        existing_entities=json.dumps(existing_entities, indent=2, default=lambda o: str(o)),
    )
    user_prompt = EXTRACTION_PROMPT.format(
        text=chunk_text
    )
    try:
        response = client.responses.parse(
            instructions=system_prompt,
            model="gpt-5",
            input=user_prompt,
            text_format=ExtractionResponse,
        )
        
        extracted_data = response.output_parsed
        return {
            "entities": [e.model_dump() for e in extracted_data.entities],
            "relationships": [r.model_dump() for r in extracted_data.relationships]
        }
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {"entities": [], "relationships": []}

# end::extraction[]

# tag::linking[]
def link_entities_to_chunk(chunk_id, entities, relationships):
        with driver.session() as session:
            session.run("""
            MATCH (c:Chunk {id: $chunk_id})
            SET c:Processed
            WITH c
            UNWIND $entities AS entity
            MERGE (e:$(entity.type) {id:entity.id})
            ON CREATE SET e += coalesce(entity.attributes,{})
            SET e:__Entity__
            MERGE (c)-[:HAS_ENTITY]->(e)
            """, {"chunk_id": chunk_id, "entities": entities})

            for rel in relationships:
                try:
                    session.run("""
                    UNWIND $relationships AS rel
                    MATCH (source:__Entity__ {id: rel.sourceId})
                    WHERE rel.sourceType IN labels(source)
                    MATCH (target:__Entity__ {id: rel.targetId})
                    WHERE rel.targetType in labels(target)
                    MERGE (source)-[r:$(rel.type)]->(target)
                    ON CREATE SET r += coalesce(rel.attributes,{})
                    """, {"relationships": [rel]})
                except Exception as e:
                    print(f"Error for relationshiop {rel} - {e}")

# end::linking[]

# tag::process[]
def process_all_chunks():
        # Load existing entities
        entity_results = driver.execute_query("""
            MATCH (e:__Entity__) 
            RETURN e.id as id, properties(e) as attributes,
                   [l in labels(e) WHERE l <> '__Entity__'][0] as type
        """)
        entities = [record.data() for record in entity_results.records]

        # Load chunks to process
        chunk_results = driver.execute_query("""
            MATCH (c:Chunk&!Processed) 
            RETURN c.id AS id, c.text AS text // LIMIT 2
        """)
        chunks = [{"id": record["id"], "text": record["text"]} for record in chunk_results.records]

        for chunk in tqdm(chunks, desc="Processing chunks"):
            results = extract_entities(chunk["text"], SCHEMA, entities)
            link_entities_to_chunk(chunk["id"], results['entities'], results['relationships'])
# end::process[]

def get_chunk_count():
        result = driver.execute_query("MATCH (c:Chunk) RETURN count(c) AS count")
        return result.records[0]["count"]

if __name__ == "__main__":
        print(f"Starting entity extraction for {get_chunk_count()} chunks...")
        process_all_chunks()
        print("Entity extraction complete!")
        driver.close()