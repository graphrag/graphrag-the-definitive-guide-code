uv run --env-file .env extraction.py ../../data/ch10/articles_update.csv
uv run --env-file .env entity_resolution.py
uv run --env-file .env initial_summarization.py
uv run --env-file .env incremental_update.py