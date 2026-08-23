export OPENAI_API_KEY="your_openai_api_key_here"

. venv/bin/activate
# python -m ensurepip
uv pip install -r requirements.txt
uv run python rag_ingest.py ../../data/ch02/
uv run streamlit run rag_app.py 
