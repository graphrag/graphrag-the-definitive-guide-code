# export OPENAI_API_KEY="your_openai_api_key_here"

. .venv/bin/activate
# python -m ensurepip
uv pip install -r requirements.txt
uv run python lexical_graph.py ../../data/ch02
uv run python structured_import.py ../../data/ch02/structured
uv run python entity_extraction.py
#uv run streamlit run rag_app.py 
