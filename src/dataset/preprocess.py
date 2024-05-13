import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
def build_vocab(data, spacy_model="fr_core_news_sm"):
    counter = Counter()
    nlp = spacy.load(spacy_model)
    for text in data:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        counter.update(tokens)
    vocab = {token: index for index, (token, _) in enumerate(counter.most_common())}
    return vocab, nlp

def collate_fn(batch):
    # Separate inputs and targets
    inputs, targets = zip(*batch)

    # Pad sequences to the maximum length in the batch
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return padded_inputs, targets_tensor
