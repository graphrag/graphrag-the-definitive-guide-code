# GraphRAG: The Definitive Guide — Code & Data

Companion code repository for the O'Reilly book **GraphRAG: The Definitive Guide** by Stephen Chin, Michael Hunger, and Jesús Barrasa.

> ### [Read the book on O'Reilly →](https://www.oreilly.com/library/view/graphrag-the-definitive/9798341630147/)

GraphRAG is any generative AI system that uses a knowledge graph to help store and retrieve the information an LLM reasons over. The book starts from the problem — LLMs hallucinate, and plain vector RAG cannot answer questions whose evidence is spread across documents — and works up from a basic ChromaDB chatbot to graph-backed retrieval, agentic tool use, persistent agent memory, enriched graph layers such as community summaries, and evaluation. A running case study (OMG Consulting, a fictional consultancy that wants to staff projects from CVs, project reports, and HR data) carries the worked chapters, while the pattern-catalog chapters generalise what those chapters do into a reusable vocabulary of construction, model, and retrieval patterns.

This repository contains the `code/` and `data/` folders referenced by the book. Not every chapter has code — conceptual and pattern chapters are marked with a dash below.

## Table of contents

| # | Chapter | What it covers | Code |
|---|---------|----------------|------|
| 1 | Introduction to GraphRAG | Why LLMs hallucinate, what a knowledge graph is, the three core GraphRAG architectures, and benchmark/case-study evidence. | — |
| 2 | A Basic RAG Assistant | OMG Consulting's first assistant: OpenAI embeddings + ChromaDB + Streamlit, and exactly where flat vector RAG breaks down. | [`code/ch02`](code/ch02) |
| 3 | Better Answers with GraphRAG | Rebuilding the same assistant on a hybrid Neo4j graph: lexical graph, structured import, entity extraction, then graph-enhanced retrieval. | [`code/ch03`](code/ch03) |
| 4 | Knowledge Graph Concepts | Knowledge graphs from five role perspectives, the three-layer KG stack, and one aircraft/airport ontology expressed five different ways. | [`code/ch04`](code/ch04) |
| 5 | Basic GraphRAG Construction Patterns | The seven basic construction patterns, from document representation through relationship extraction. | — |
| 6 | Basic GraphRAG Model and Retrieval Patterns | Three graph-model patterns (lexical, entity, hybrid) and three retrieval patterns built on top of them. | — |
| 7 | Knowledge Graph Construction | Applying construction patterns to a multi-source enterprise scenario: metadata-only vs materialised sources, ontology-guided extraction, entity resolution. | — |
| 8 | Agentic GraphRAG | Tool calling, MCP, the Neo4j MCP server, custom vector/fulltext/recommendation tools, and an autonomous agent loop. | [`code/ch08`](code/ch08) |
| 9 | Agent Memory and Context Graphs | A context graph holding short-term conversation, long-term domain knowledge, and reasoning traces, built with `neo4j-agent-memory`. | [`code/ch09`](code/ch09) |
| 10 | Enriched Knowledge Graphs | Microsoft-style query-focused summarization over a news corpus: extraction, entity resolution, community detection, community summaries, then local and global search. | [`code/ch10`](code/ch10) |
| 11 | Advanced GraphRAG Patterns | Fifteen advanced construction, model, and retrieval patterns, including six memory patterns, plus a decision framework. | — |
| 12 | Evaluating GraphRAG | The four-layer evaluation pyramid, plus two working benchmarks: agent end-to-end scoring and KG construction scoring. | [`code/ch12`](code/ch12) |
| 13 | Open Problems, Pro Tips, and Next Steps | What is still unsolved, practical advice, and where to go next. | — |

## Authors

- **Stephen Chin** — VP of Developer Relations at Neo4j, conference chair of the LF AI & Data Foundation, and author of numerous O'Reilly, Apress, and McGraw Hill titles. LinkedIn: https://www.linkedin.com/in/steveonjava/
- **Michael Hunger** — 35+ years in software development and 15 years on the open source Neo4j graph database, most recently leading Product Innovation and Developer Product Strategy, currently focused on generative AI, GraphRAG, and developer experience. LinkedIn: https://www.linkedin.com/in/jexpde
- **Jesús Barrasa** — Field CTO for AI at Neo4j, co-author of *Building Knowledge Graphs* (O'Reilly, 2023), cohost of [Going Meta](https://goingmeta.live/), and holder of a Ph.D. in AI/Knowledge Representation. LinkedIn: https://www.linkedin.com/in/jbarrasa/

## Repository layout

```
code/
  ch02/  Basic RAG: ChromaDB ingest + Streamlit chat app
  ch03/  GraphRAG: lexical graph, structured import, entity extraction,
         graph refinement, retrievers, Streamlit app, quick eval
  ch04/  Ontology artifacts only (no pipeline): SKOS + OWL Turtle,
         a Pydantic schema, a neo4j-graphrag JSON schema, a Palantir JSON export
  ch08/  Agentic GraphRAG: agent loop, MCP client, vector/fulltext/
         recommendation tools, index creation, Streamlit app
  ch09/  Agent memory: OMG graph adoption, memory settings/IO,
         memory-aware agent + Streamlit app, example Cypher queries
  ch10/  Enrichment pipeline: extraction, entity resolution, summarization,
         community detection & summaries, incremental update, local/global search
  ch12/  Evaluation: agent benchmark, KG construction benchmark,
         ontology extractor, benchmark datasets
data/
  ch02/  OMG Consulting corpus — 27 CV markdown files, 40 project reports,
         an OMG company overview, and 4 structured CSVs (employees, projects,
         skills, project participation). Used by chapters 2, 3, 8, 9 and 12.
  ch03/  retriever_outputs.adoc — transcript of retriever output shown in the book
  ch08/  project_description_embedding.csv — precomputed embeddings
  ch10/  articles_initial.csv and articles_update.csv — the news corpus
```

## Running the code examples

### Prerequisites

- **Python 3.10 or newer.** The repository does not pin a version; 3.10 is the floor required by the current `neo4j` Python driver (v6.x).
- **[uv](https://docs.astral.sh/uv/)** — every `run_*.sh` script in this repo uses `uv`, and `uvx` must be on your `PATH` for the MCP clients in Chapter 8 and Chapter 12, which both launch the Neo4j MCP server with `uvx neo4j-mcp-server` (`uvx` downloads the `neo4j-mcp-server` package on first use, so the first run needs network access).
- An **OpenAI API key**.
- A **Neo4j instance** for chapters 3, 8, 9, 10 and 12 (see [Setting up Neo4j Aura](#setting-up-neo4j-aura)). Chapter 2 needs no database — it uses a local ChromaDB store.

### Installing dependencies

Dependencies are declared per chapter, and the mechanism differs by chapter:

| Chapter | How to install |
|---------|----------------|
| ch02, ch03, ch08, ch09, ch10 | `uv pip install -r requirements.txt` inside the chapter folder |
| ch12 | No requirements file; `run_eval.sh` passes deps inline via `uv run --with openai --with neo4j --with python-dotenv --with pandas --with pydantic --with mcp` |
| ch04 | No runtime dependencies beyond `pydantic` if you want to import `onto-as-pydantic.py`; the other files are ontology artifacts |

A single shared virtual environment works for all chapters:

```bash
uv venv
source .venv/bin/activate
```

### Environment variables

Create a `graphrag-book.env` file in the chapter folder you are running — copy `graphrag-book.env.example` from the repository root and fill it in. Every chapter uses that same filename, either through `load_dotenv("graphrag-book.env")` or through `uv run --env-file graphrag-book.env`. The `.gitignore` already excludes `*.env`, so your filled-in copy will not be committed.

```bash
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
```

Optional variables used by individual chapters:

| Variable | Used by | Default in code |
|----------|---------|-----------------|
| `NEO4J_DATABASE` | ch08, ch09 | `neo4j` |
| `NEO4J_READ_ONLY` | ch08 | `true` |
| `OPENAI_MODEL` | ch03, ch08, ch09 | `gpt-5` |
| `JUDGE_MODEL` | ch12 | `gpt-5` |
| `TARGET_MODEL` | ch12 | `gpt-5.4-mini` |

Chapters 10 and 12 never call `load_dotenv()` themselves — their shell scripts pass `uv run --env-file graphrag-book.env`, so the file is mandatory there rather than optional.

All paths in the scripts are relative, and the shell scripts do not `cd`. **Always run them from inside the chapter folder.**

### Running a chapter

Each chapter that needs setup ships a shell script; the interactive apps are Streamlit.

```bash
# Chapter 2 — basic RAG (no database required)
cd code/ch02
uv pip install -r requirements.txt
uv run --env-file graphrag-book.env python rag_ingest.py ../../data/ch02/
uv run --env-file graphrag-book.env streamlit run rag_app.py

# Chapter 3 — build the graph, then chat against it
cd code/ch03
bash run_graphrag.sh                       # lexical graph + structured import + entity extraction
uv run streamlit run app_graphrag.py

# Chapter 8 — agentic GraphRAG (also the base graph for ch09 and ch12)
cd code/ch08
bash run_import.sh                         # structured import + vector index
uv run streamlit run app_agent.py

# Chapter 9 — agent memory (requires the ch08 graph)
cd code/ch09
bash run_setup.sh
uv run streamlit run app_agent_memory.py

# Chapter 10 — enrichment pipeline and searches
cd code/ch10
bash run_ingestion.sh
bash run_searches.sh
bash run_incremental_update.sh             # optional: incremental re-summarization

# Chapter 12 — evaluation benchmarks
cd code/ch12
bash run_import.sh                         # loads the ch08 graph the agent benchmark scores against
bash run_eval.sh
```

## Setting up Neo4j Aura

Aura is Neo4j's managed cloud service and has a free tier that is enough for the book's examples.

1. Go to **[console.neo4j.io](https://console.neo4j.io)** and sign up (Google/GitHub/email — no credit card needed for the free tier).
2. Choose **Create instance** and pick the **AuraDB Free** instance type, then a region near you. Provisioning takes a couple of minutes.
3. When the instance is created, Aura shows the generated password and offers a **credentials file** (`Neo4j-<instance-id>-Created-<date>.txt`) to download. **Download it now — the password is displayed exactly once and cannot be recovered.** If you lose it, you must reset the password from the instance's settings. Copy its values into your `graphrag-book.env` file.
4. Wait for the instance status to turn **Running**.

The credentials file maps directly onto the environment variables the code expects in `graphrag-book.env`:

| Credentials file field | `graphrag-book.env` variable |
|------------------------|-----------------|
| `NEO4J_URI` (`neo4j+s://<id>.databases.neo4j.io`) | `NEO4J_URI` |
| `NEO4J_USERNAME` (always `neo4j`) | `NEO4J_USERNAME` |
| `NEO4J_PASSWORD` | `NEO4J_PASSWORD` |
| `NEO4J_DATABASE` (always `neo4j` on Aura) | `NEO4J_DATABASE` (optional) |
| `OPENAI_API_KEY` | `sk-xxx` add your own OpenAI API key |

Two things to keep in mind about the free tier: it has **size limits** on the number of nodes and relationships (the current figures are shown in the console and on the [AuraDB FAQ](https://neo4j.com/cloud/platform/aura-graph-database/faq/) — Neo4j's own pages have disagreed on the number, so check the console), and **free instances are paused automatically after a period of inactivity** and deleted if left paused. The book's datasets are small and fit comfortably.

If you would rather run locally, [Neo4j Desktop](https://neo4j.com/download/) or the official Docker image work equally well — set `NEO4J_URI=bolt://localhost:7687` and your local credentials. Chapter 10's `utils.py` already defaults to `bolt://localhost:7687` / `neo4j` / `password`.

## Getting help

- Book page: <https://www.oreilly.com/library/view/graphrag-the-definitive/9798341630147/>
- Issues with the code: open an issue on this repository.
- Neo4j community: <https://community.neo4j.com>
