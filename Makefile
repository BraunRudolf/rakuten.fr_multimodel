.PHONY: setup
setup:
	poetry install || (echo "Error installing dependencies" && exit 1)
	poetry run python -m spacy download fr_core_news_sm || (echo "Error downloading spaCy model" && exit 1)
