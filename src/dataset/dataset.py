import os
from typing import Callable, Dict, List, Tuple, Union

import torch
from sqlalchemy import MetaData, create_engine, select
from torch.utils.data import DataLoader, Dataset
from torchvision import io


class RakutenTextDataset(Dataset):
    def __init__(
        self,
        db_url: str,
        table_name: str,
        mapping_table_name: str,
        text_column: str,
        label_column: str,
        mapping_column: str,
        vocab: dict,
        spacy_model,
        indices: List[int],
        preprocessing_pipeline: List[Callable] = [],
    ):

        self.db_url = db_url
        self.table_name = table_name
        self.mapping_table_name = mapping_table_name
        self.text_column = text_column
        self.label_column = label_column
        self.mapping_column = mapping_column
        self.vocab = vocab
        self.nlp = spacy_model
        # self.tokenizer = spacy_model.tokenizer
        self.preprocessing_pipeline = preprocessing_pipeline or []
        self.indices = indices
        self.metadata = MetaData()
        self.engine = self.connect_to_db()
        self.conn = self.engine.connect()
        # self.table = Table(self.table_name, self.metadata, autoload_with=self.engine)

    def connect_to_db(self):
        engine = create_engine(self.db_url)
        self.metadata.reflect(engine)
        self.table = self.metadata.tables[self.table_name]
        self.mapping_table = self.metadata.tables[self.mapping_table_name]

        return engine

    def preprocess_text(self, text):
        if self.preprocessing_pipeline is None:
            return self.nlp(text)
            return [token.text for token in self.tokenizer(text)]

        for func in self.preprocessing_pipeline:
            text = func(text)
        return self.nlp(text)
        # return [token.text for token in self.tokenizer(text)]

    def text_to_tensor(self, doc):
        # preprocessed_text = self.preprocess_text(doc)
        tokens = [token.text for token in doc]
        tensor = [self.vocab.get(token, self.vocab.get("<unk>")) for token in tokens]

        # Check for None values in tensor
        if any(val is None for val in tensor):
            raise ValueError("Vocabulary lookup returned None for some tokens.")
        return torch.tensor(tensor, dtype=torch.long)
        tensor = [self.vocab[token] for token in preprocessed_text if token in self.vocab]
        return torch.tensor(tensor, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    #  SQl on the fly implementation
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[str, int]]:
        # INFO: Prove of concept, alternative load complete text data in memory and access from there
        index = self.indices[idx]
        with self.engine.connect() as conn:
            stmt = (
                select(self.table.c[self.text_column], self.mapping_table.c[self.label_column])
                .outerjoin(
                    self.mapping_table,
                    self.table.c[self.mapping_column] == self.mapping_table.c[self.mapping_column],
                )
                .where(self.table.c.id == index)
            )
            result = conn.execute(stmt).fetchone()
        text, label = result
        doc = self.preprocess_text(text)
        text_tensor = self.text_to_tensor(doc)
        return text_tensor, label


class RakutenImageDataset(Dataset):
    def __init__(self, image_folder, image_names, labels=None, transform=None):
        self.image_folder = image_folder
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.image_paths = [
            os.path.join(self.image_folder, image_name) for image_name in self.image_names
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = io.read_image(self.image_paths[idx])
        image = image.float()
        if self.transform:
            image = self.transform(image)
        if self.labels:
            return image, self.labels[idx]
        return image


class RakutenFusionDataset(Dataset):
    def __init__(self, text_dataset, image_dataset):
        self.text_dataset = text_dataset
        self.image_dataset = image_dataset

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, idx):

        text, label_text = self.text_dataset[idx]
        image, label_image = self.image_dataset[idx]

        if label_text != label_image:
            raise Exception()
        return text, image, label_text
