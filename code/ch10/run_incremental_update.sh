uv run --env-file graphrag-book.env extraction.py ../../data/ch10/articles_update.csv
uv run --env-file graphrag-book.env entity_resolution.py
uv run --env-file graphrag-book.env initial_summarization.py
uv run --env-file graphrag-book.env incremental_update.py