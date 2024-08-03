# rakuten.fr_multimodel
## requirements 
- python 3.12.0
- poetry 1.8.2
- Datafiles: https://challengedata.ens.fr/participants/challenges/35/
    - last checked: 2024-08-03
    - X_train_update.csv
    - y_train_CVw08PX.csv
    - images.zip

## enviroment varialbes
DB_SERVER_URL=sqlite:////database.db

MLFLOW_SERVER_URI=http://127.0.0.1:5000

MLFLOW_EXPERIMENT_NAME

VOCAB_PATH=vocab/

## install dependencies with poetry
`make install`

## train the model
if using local mlflow run
`make train_local_mlflow`

otherwise 
`make train`

## To-Do
- [ ] Commenting
- [ ] README.md
- [ ] Unit-Test
- [ ]  Image model/training/...
- [ ] Text-Preprocessing

...
