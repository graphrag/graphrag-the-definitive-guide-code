# graph_refinement.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv("graphrag-book.env")

# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Normalize and merge similar entity names
def normalize_entity_names(label, property_name="name"):
    with driver.session() as session:
        # Find potential duplicates based on string similarity
        duplicates = session.run(f"""
        MATCH (a:{label}), (b:{label})
        WHERE a <> b AND a.{property_name} <> b.{property_name}
        AND apoc.text.levenshteinSimilarity(toLower(a.{property_name}), toLower(b.{property_name})) > 0.8
        RETURN a.{property_name} AS name1, b.{property_name} AS name2, 
               apoc.text.levenshteinSimilarity(a.{property_name}, b.{property_name}) AS similarity
        ORDER BY similarity DESC
        """).data()
        
        # Display potential duplicates for manual review
        print(f"Potential duplicate {label} entities:")
        for dup in duplicates:
            print(f"  {dup['name1']} <-> {dup['name2']} (similarity: {dup['similarity']:.2f})")
        
        # Optional: Implement automatic merging logic here
        # This example shows manual decision logic
        for dup in duplicates:
            decision = input(f"Merge '{dup['name1']}' and '{dup['name2']}'? (y/n/keep1/keep2): ")
            
            if decision.lower() == 'y':
                # Merge with combined name
                combined_name = f"{dup['name1']} ({dup['name2']})"
                session.run(f"""
                MATCH (a:{label} {{{property_name}: $name1}}), (b:{label} {{{property_name}: $name2}})
                CALL apoc.merge.nodes([a, b], {{{property_name}: $combined_name}})
                YIELD node
                RETURN node
                """, {"name1": dup['name1'], "name2": dup['name2'], "combined_name": combined_name})
                print(f"  Merged as '{combined_name}'")
            
            elif decision.lower() == 'keep1':
                # Keep first name, merge nodes
                session.run(f"""
                MATCH (a:{label} {{{property_name}: $name1}}), (b:{label} {{{property_name}: $name2}})
                CALL apoc.merge.nodes([a, b], {{{property_name}: $name1}})
                YIELD node
                RETURN node
                """, {"name1": dup['name1'], "name2": dup['name2']})
                print(f"  Merged as '{dup['name1']}'")
            
            elif decision.lower() == 'keep2':
                # Keep second name, merge nodes
                session.run(f"""
                MATCH (a:{label} {{{property_name}: $name1}}), (b:{label} {{{property_name}: $name2}})
                CALL apoc.merge.nodes([a, b], {{{property_name}: $name2}})
                YIELD node
                RETURN node
                """, {"name1": dup['name1'], "name2": dup['name2']})
                print(f"  Merged as '{dup['name2']}'")
            
            else:
                print("  No action taken")

# Add missing project-technology relationships based on chunk mentions
def infer_project_technologies():
    with driver.session() as session:
        # Find technologies mentioned in project documents but not explicitly linked
        inferred_techs = session.run("""
        MATCH (p:Project)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(t:Technology)
        WHERE NOT (p)-[:USED_TECHNOLOGY]->(t)
        RETURN p.name AS project, t.name AS technology, count(c) AS mentions
        ORDER BY mentions DESC
        """).data()
        
        print(f"Found {len(inferred_techs)} potential project-technology relationships to add")
        
        # For high-confidence inferences (multiple mentions), add the relationship
        for item in inferred_techs:
            if item['mentions'] >= 2:  # Threshold for confidence
                session.run("""
                MATCH (p:Project {name: $project}), (t:Technology {name: $technology})
                MERGE (p)-[:USED_TECHNOLOGY]->(t)
                """, {"project": item['project'], "technology": item['technology']})
                print(f"  Added: {item['project']} -> {item['technology']} ({item['mentions']} mentions)")
            else:
                print(f"  Skipped: {item['project']} -> {item['technology']} (only {item['mentions']} mention)")

# Add skill hierarchies and relationships
def enhance_skill_taxonomy():
    with driver.session() as session:
        # Define skill categories and their members
        skill_taxonomy = {
            "Programming Languages": ["JavaScript", "Python", "TypeScript", "Java", "C#", "Go", "Ruby"],
            "Frontend Technologies": ["React", "React.js", "Angular", "Vue", "HTML", "CSS", "Next.js"],
            "Backend Technologies": ["Node.js", "Express", "Django", "Flask", "Spring Boot", "ASP.NET"],
            "Database Technologies": ["Neo4j", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch"],
            "Cloud Platforms": ["AWS", "Azure", "Google Cloud", "IBM Cloud"],
            "DevOps Tools": ["Docker", "Kubernetes", "Jenkins", "GitHub Actions", "CircleCI", "Terraform"],
            "AI and ML": ["Machine Learning", "Deep Learning", "Neural Networks", "NLP", "Computer Vision", "LangChain"]
        }
        
        # Create category nodes and relationships
        for category, skills in skill_taxonomy.items():
            # Create category
            session.run("""
            MERGE (c:SkillCategory {name: $category})
            """, {"category": category})
            
            # Link skills to category
            session.run("""
            MATCH (c:SkillCategory {name: $category})
            UNWIND $skills AS skill_name
            MATCH (s:Skill)
            WHERE s.name = skill_name OR s.name CONTAINS skill_name OR skill_name CONTAINS s.name
            MERGE (s)-[:IN_CATEGORY]->(c)
            """, {"category": category, "skills": skills})
            
        print("Enhanced skill taxonomy with categories")

# Main execution
if __name__ == "__main__":
    print("Starting graph refinement...")
    
    print("\nNormalizing Technology names...")
    normalize_entity_names("Technology")
    
    print("\nNormalizing Skill names...")
    normalize_entity_names("Skill")
    
    print("\nInferring project-technology relationships...")
    infer_project_technologies()
    
    print("\nEnhancing skill taxonomy...")
    enhance_skill_taxonomy()
    
    print("\nRefinement complete!")
    driver.close()
