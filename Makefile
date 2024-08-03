.PHONY: setup
setup:
	poetry install || (echo "Error installing dependencies" && exit 1)
	mkdir -p vocab
	poetry run python -m spacy download fr_core_news_sm || (echo "Error downloading spaCy model" && exit 1)
	poetry run python setup_scripts/save_dataset_to_db.py
	poetry run python setup_scripts/setup_images.py

.PHONY: train_local_mlflow
train_local_mlflow:
	poetry run bash setup_scripts/run_local_mlflow.sh

.PHONY: train
train:
	poetry run python -m src.main
