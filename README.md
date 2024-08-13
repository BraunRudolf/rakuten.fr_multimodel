# rakuten.fr_multimodel
## Project Overview
This project aims to develop a robust multimodal classification model for predicting product type codes based on product titles, images, and potentially additional descriptions. The goal is to improve product categorization accuracy and efficiency for Rakuten France. As a proof of concept, text data stored in a database to facilitate scalability and efficient data management. MLflow is integrated to track experiments, monitor model performance, and facilitate reproducibility.

### Problem
Accurate and scalable product type classification is crucial for e-commerce platforms like Rakuten. Manual and rule-based methods are often insufficient due to the vast and diverse product catalog.

### Solution
A multimodal fusion model is implemented to leverage both textual and visual information for enhanced classification performance. MLflow is integrated for experiment tracking, model management, and reproducibility.

## Dataset
### Description
The dataset consists of product listings with corresponding product type codes from Rakuten France. Each data point includes product title, image, and optional description.
Preprocessing: Data preprocessing includes image resizing, text cleaning, and tokenization.

## requirements
- python 3.12.0
- poetry 1.8.2
- Datafiles: https://challengedata.ens.fr/participants/challenges/35/
    - last checked: 2024-08-03
    - X_train_update.csv
    - y_train_CVw08PX.csv
    - images.zip
# Add sample data

## enviroment varialbes
DB_SERVER_URL=sqlite:///database.db

MLFLOW_SERVER_URI=http://127.0.0.1:5000

MLFLOW_EXPERIMENT_NAME

VOCAB_PATH=vocab/

IMAGE_ZIP=images.zip

IMAGE_FOLDER=images

IMAGE_TRAIN_FOLDER=image_train

NUM_OF_WORKERS=0
PIN_MEMORY=False

ID_COLUMN=id

IMAGE_COLUMN=image_name

TABLE_NAME=rakuten_products

MAPPING_TABLE_NAME=prdtypecode_label_mapping

TEXT_COLUMN=text

LABLE_COLUMN=label

MAPPING_COLUMN=prdtypecode

## install dependencies with poetry
`make install`

## train the model
if using local mlflow run
`make train_local_mlflow`
Since the sqlite database is used the number of workers is set to 0, this can result in a slow training time.


otherwise 
`make train`

## To-Do
- [ ] Commenting
- [ ] README.md
- [ ] Unit-Test
- [ ]  Image model/training/...
- [ ] Text-Preprocessing

...
