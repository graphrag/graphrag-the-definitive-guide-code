#. .venv/bin/activate
# python -m ensurepip
uv pip install -r requirements.txt
uv run --env-file graphrag-book.env python ../ch03/structured_import.py ../../data/ch02/structured
uv run --env-file graphrag-book.env python create_index.py ../../data/ch08
