import os
import pandas as pd
from typing import Union, List
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math


class DataImporter:
    #TODO:change target_col to feature_col
    def __init__(self, target_col: Union[str, List[str]], label_col: str,
                 datapath: str = "data/preprocessed", filename: str = 'dataset.csv'):
        self.filepath = os.path.join(datapath, filename)
        self.datapath = datapath
        self.filename = filename
        self.target_col = target_col
        self.label_col = label_col

    def load_data(self):
        try:
            data = pd.read_csv(self.filepath, index_col=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"File data['target'] = data['target'].fillna(''not found at {self.filepath}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"No data found in {self.filepath}")
        except Exception as e:
            raise e

        for col in self.target_col:
            data[col] = data[col].fillna('')

        if isinstance(self.target_col, list):
            data['target'] = data[self.target_col[0]].astype(str)
            for col in self.target_col[1:]:
                data['target'] += " " + data[col].astype(str)
        else:
            data['target'] = data[self.target_col].astype(str)

        data['label'] = data[self.label_col]

        return data[['target', 'label']]

    def split_train_test(self, df, samples_per_class=600):

        grouped_data = df.groupby("label")

        X_train_samples = []
        X_test_samples = []

        for _, group in grouped_data:
            samples = group.sample(n=samples_per_class, random_state=42)
            X_train_samples.append(samples)

            remaining_samples = group.drop(samples.index)
            X_test_samples.append(remaining_samples)

        X_train = pd.concat(X_train_samples)
        X_test = pd.concat(X_test_samples)

        X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_test = X_test.sample(frac=1, random_state=42).reset_index(drop=True)

        y_train = X_train["label"]
        X_train = X_train["target"]

        y_test = X_test["label"]
        X_test = X_test["target"]

        val_samples_per_class = 50

        grouped_data_test = pd.concat([X_test, y_test], axis=1).groupby("label")

        X_val_samples = []
        y_val_samples = []

        for _, group in grouped_data_test:
            samples = group.sample(n=val_samples_per_class, random_state=42)
            X_val_samples.append(samples["target"])
            y_val_samples.append(samples["label"])

        X_val = pd.concat(X_val_samples)
        y_val = pd.concat(y_val_samples)

        X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)

        assert X_train.isna().sum() == 0
        assert X_val.isna().sum() == 0
        assert X_test.isna().sum() == 0
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class ImagePreprocessor:
    def __init__(self, filepath="data/preprocessed/image_train"):
        self.filepath = filepath

    def preprocess_images_in_df(self, df):
        df["image_path"] = (
            f"{self.filepath}/image_"
            + df["imageid"].astype(str)
            + "_product_"
            + df["productid"].astype(str)
            + ".jpg"
        )


class TextPreprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(
            stopwords.words("french")
        )
    def preprocess_text(self, text:str):

        if isinstance(text, float) and math.isnan(text):
            return ""
        
        # Supprimer les balises HTML
        text = BeautifulSoup(text, "html.parser").get_text()

        # Supprimer les caractères non alphabétiques
        text = re.sub(r"[^a-zA-Z]", " ", text)

        # Tokenization
        words = word_tokenize(text.lower())

        # Suppression des stopwords et lemmatisation
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(filtered_words[:100])

    def preprocess_text_in_df(self, df:Union[pd.DataFrame,pd.Series], columns:List[str]=['']):
        if isinstance(df, pd.DataFrame):
            for column in columns:
                df[column] = df[column].apply(self.preprocess_text)
            return df

        if isinstance(df, pd.Series):
            return df.apply(self.preprocess_text)

