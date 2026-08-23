import sys
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv("graphrag-book.env")

# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# tag::structured_import[]

def setup_domain_schema():
    with driver.session() as session:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Client) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technology) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.id IS UNIQUE"
        ]
        
        for constraint in constraints:
            session.run(constraint)

import pandas as pd

def import_people(base_path):
    people_df = pd.read_csv(base_path + "/employees.csv")
    
    with driver.session() as session:
        # Create person nodes and management relationships
        session.run("""
        UNWIND $people AS person
        MERGE (p:Person {id: person.employee_id})
        SET p:__Entity__
        SET p += person {.full_name, .role, .department, .email, .phone, .office_location, .expertise, .seniority},
            p.joined = date(person.join_date)
        WITH p, person WHERE person.manager_id <> ''
        MERGE (manager:Person {id: person.manager_id})
        MERGE (manager)-[:MANAGES]->(p)
        """, {"people": people_df.to_dict("records")})
            
        print(f"Imported {len(people_df)} people")

def import_skills(base_path):
    skills_df = pd.read_csv(base_path + "/consultant_skills.csv")
    
    with driver.session() as session:
        session.run("""
        UNWIND $skills AS skill
        MATCH (p:Person {id: skill.employee_id})
        MERGE (s:Skill {id: skill.skill_name})
        ON CREATE SET s.name = skill.skill_name
        // SET s:__Entity__
        MERGE (p)-[r:HAS_SKILL]->(s)
        SET r += skill {.skill_category, .proficiency_level, .certification, .skill_description, .project_acquired},
            r.category = skill.skill_category,
            r.proficiency = skill.proficiency_level,
            r.acquisition_date = date(skill.acquisition_date),
            r.last_used = CASE skill.last_used_date WHEN 'Active' THEN null ELSE date(skill.last_used_date) END,
            r.description = skill.skill_description,
            r.project_acquired = skill.project_acquired
        """, {"skills": skills_df.to_dict("records")})
        
        print(f"Imported {len(skills_df)} skill relationships")

# end::structured_import[]

def import_projects_clients(base_path):
    projects_df = pd.read_csv(base_path + "/projects.csv")
    
    with driver.session() as session:
        session.run("""
        UNWIND $projects AS project
        MERGE (p:Project {id: project.project_id})
        SET p:__Entity__
        SET p += project {.description, .status, .project_type, .budget_range, .location},
            p.name = project.project_name,
            p.start_date = date(project.start_date),
            p.end_date = case when project.end_date ='Active' then null else date(project.end_date) end
            
        WITH p, project
        MERGE (c:Client {id: project.client_name})
        SET c:__Entity__
        SET c.name = project.client_name
        MERGE (p)-[:FOR_CLIENT]->(c)
        
        WITH p, project, c
        MERGE (i:Industry {id: project.industry})
        ON CREATE SET i.name = project.industry
        MERGE (c)-[:IN_INDUSTRY]->(i)
        """, {"projects": projects_df.to_dict("records")})
        
        print(f"Imported {len(projects_df)} projects and their clients")

def import_project_assignments(base_path):
    assignments_df = pd.read_csv(base_path + "/project_participation.csv")
    
    with driver.session() as session:
        session.run("""
        UNWIND $assignments AS assignment
        MATCH (person:Person {id: assignment.employee_id})                    
        MATCH (project:Project {id: assignment.project_id})
        MERGE (person)-[r:WORKED_ON]->(project)
        SET r += assignment {.role, .utilization_percentage, .key_contributions},
            r.start_date = date(assignment.start_date),
            r.end_date = case when assignment.end_date ='Active' then null else date(assignment.end_date) end
        """, {"assignments": assignments_df.to_dict("records")})
        
        print(f"Imported {len(assignments_df)} project assignments")

def import_project_technologies(base_path):
    # Create technology relationships from skills data
    skills_df = pd.read_csv(base_path + "/consultant_skills.csv")
    
    with driver.session() as session:
        # Create Technology nodes from skills and link to projects where acquired
        session.run("""
        UNWIND $skills AS skill
        WITH skill WHERE skill.project_acquired IS NOT NULL AND skill.project_acquired <> ''
        MERGE (t:Technology {id: skill.skill_name})
        ON CREATE SET t.name = skill.skill_name, t.category = skill.skill_category
        // SET t:__Entity__
        WITH * 
        MATCH (s:Skill {id: skill.skill_name})
        MERGE (s)-[:SKILL_FOR]->(t)
        WITH distinct t, skill
        MATCH (project:Project {id: skill.project_acquired})
        MERGE (project)-[r:USED_TECHNOLOGY]->(t)
        SET r.context = 'skill_acquired'
        """, {"skills": skills_df.to_dict("records")})
        
        print(f"Imported technology relationships from skills data")

# tag::structured_import[]

if __name__ == "__main__":
    base_path = sys.argv[1] if len(sys.argv) > 1 else "./data/ch02/structured"
    print("Setting up domain schema...")
    setup_domain_schema()
    
    print("Importing structured data...")
    import_people(base_path)
    import_skills(base_path)

# end::structured_import[]
    import_projects_clients(base_path)
    import_project_assignments(base_path)
    import_project_technologies(base_path)
    
    print("Structured data import complete!")
    driver.close()
