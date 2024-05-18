import os
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import json

def build_vocab(data, spacy_model="fr_core_news_sm", save_dir=None):
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
