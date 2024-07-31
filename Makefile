install:
	poetry install
	poetry run python -m spacy downlaod fr_core_news_sm
