import json
import os
from collections import Counter
from typing import Optional

import pandas as pd
import spacy
import torch
from sklearn.model_selection import train_test_split
from sqlalchemy import MetaData, Table, create_engine, select
from torch.nn.utils.rnn import pad_sequence


# NOTE: Is this realy preprocessing?
def build_vocab(data: list, spacy_model="fr_core_news_sm", save_dir=None):
    counter = Counter()
    nlp = spacy.load(spacy_model)
    for text in data:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        counter.update(tokens)
    vocab = {token: index for index, (token, _) in enumerate(counter.most_common())}
    if save_dir:
        # Save the vocabulary dictionary as a JSON file
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)

        # Save the spacy model's vocabulary
        vocab_dir = os.path.join(save_dir, "spacy_vocab")
        nlp.vocab.to_disk(vocab_dir)
    return vocab, nlp


def load_vocab_and_nlp(save_dir, spacy_model="fr_core_news_sm"):
    # Load the vocabulary dictionary from the JSON file
    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load the spaCy model's vocabulary
    nlp = spacy.load(spacy_model)
    vocab_dir = os.path.join(save_dir, "spacy_vocab")
    nlp.vocab.from_disk(vocab_dir)

    return vocab, nlp


def collate_fn(batch):
    # Separate inputs and targets
    inputs, targets = zip(*batch)

    # Pad sequences to the maximum length in the batch
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return padded_inputs, targets_tensor


def retrieve_indices(
    db_url: str, table_name: str, id_col: str, limit: Optional[int] = None
) -> list:
    """
    params:
        limit: number of values to be retrieved (newest first)
    returns number of ids from database

    """
    engine = create_engine(db_url)

    with engine.connect() as conn:
        metadata = MetaData()
        metadata.reflect(bind=engine)
        tables = metadata.tables[table_name]

        query = select(tables.c[id_col]).order_by(tables.c[id_col].desc()).limit(limit)

        result = conn.execute(query).fetchall()

        return [t[0] for t in result]


def train_val_test_indices(
    all_indices: list,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state=42,
):
    """Function to create a set of train/val/test indices for Dataset class"""

    assert train_size + val_size + test_size == 1.0, "Size must bum to 1.0"

    # First split: train and remaining (validation + test)
    train_indices, remaining_indices = train_test_split(
        all_indices, train_size=train_size, random_state=random_state
    )

    # Calculate the proportion of the remaining data
    remaining_size = val_size + test_size
    val_proportion = val_size / remaining_size

    # Second split: validation and test from the remaining data
    val_indices, test_indices = train_test_split(
        remaining_indices, train_size=val_proportion, random_state=random_state
    )

    return train_indices, val_indices, test_indices


def retrieve_vocab_dataset(
    db_url: str, table_name: str, id_col: str, text_col: str, indices: list
) -> list:
    engine = create_engine(db_url)

    with engine.connect() as conn:
        metadata = MetaData()
        metadata.reflect(bind=engine)
        tables = metadata.tables[table_name]

        query = (
            select(tables.c[text_col])
            .filter(tables.c[id_col].in_(indices))
            .order_by(tables.c[id_col].desc())
        )

        result = conn.execute(query).fetchall()
        return [t[0] for t in result]


def retrieve_image_info(
    db_url: str,
    table_name: str,
    mapping_table_name: str,
    id_col: str,
    image_col: str,
    mapping_column: str,
    label_col: str,
    indices: list,
) -> pd.DataFrame:
    engine = create_engine(db_url)

    with engine.connect() as conn:
        metadata = MetaData()
        metadata.reflect(bind=engine)
        tables = metadata.tables[table_name]
        mapping_table = metadata.tables[mapping_table_name]

        query = (
            select(tables.c[image_col], mapping_table.c[label_col])
            .outerjoin(
                mapping_table,
                tables.c[mapping_column] == mapping_table.c[mapping_column],
            )
            .filter(tables.c[id_col].in_(indices))
            .order_by(tables.c[id_col].desc())
        )

        result = conn.execute(query).fetchall()
        # WARNING: HARDCODED Values
        return pd.DataFrame(result, columns=["image_name", "label"])
