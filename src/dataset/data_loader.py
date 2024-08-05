from torch.utils.data import DataLoader

from src.dataset.dataset import RakutenImageDataset, RakutenTextDataset
from src.dataset.preprocess import collate_fn, retrieve_image_info


def create_text_dataloaders(
    db_url,
    table_name,
    mapping_table_name,
    text_column,
    label_column,
    mapping_column,
    vocab,
    spacy_model,
    train_indices,
    val_indices,
    test_indices,
    batch_size,
    num_workers,
    pin_memory,
):
    """
    Function to create a set of train/val/test dataloaders for text data
    parameters:
        db_url: str
        table_name: str
        mapping_table_name: str
        text_column: str
        label_column: str
        mapping_column: str
        vocab: dict
        spacy_model
        train_indices: list
        val_indices: list
        test_indices: list
        batch_size: int
        num_workers: int
        pin_memory: bool
    """

    # Creating datasets
    train_dataset = RakutenTextDataset(
        db_url,
        table_name,
        mapping_table_name,
        text_column,
        label_column,
        mapping_column,
        vocab,
        spacy_model,
        train_indices,
    )
    val_dataset = RakutenTextDataset(
        db_url,
        table_name,
        mapping_table_name,
        text_column,
        label_column,
        mapping_column,
        vocab,
        spacy_model,
        val_indices,
    )
    test_dataset = RakutenTextDataset(
        db_url,
        table_name,
        mapping_table_name,
        text_column,
        label_column,
        mapping_column,
        vocab,
        spacy_model,
        test_indices,
    )

    # Creating dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_image_dataloaders(
    db_url,
    table_name,
    mapping_table_name,
    id_column,
    image_column,
    label_column,
    mapping_column,
    train_indices,
    val_indices,
    test_indices,
    batch_size,
    num_workers,
    pin_memory,
    image_folder="images",
    transform=None,
):
    """
    Function to create a set of train/val/test dataloaders for image data
    """
    # TODO: remove retrieve_image_info for train/val/test; retrive comliete df handel selection in RakutenImageDataset class
    train_image_info = retrieve_image_info(
        db_url,
        table_name,
        mapping_table_name,
        id_column,
        image_column,
        mapping_column,
        label_column,
        train_indices,
    )
    val_image_info = retrieve_image_info(
        db_url,
        table_name,
        mapping_table_name,
        id_column,
        image_column,
        mapping_column,
        label_column,
        val_indices,
    )
    test_image_info = retrieve_image_info(
        db_url,
        table_name,
        mapping_table_name,
        id_column,
        image_column,
        mapping_column,
        label_column,
        test_indices,
    )

    # Creating datasets
    # WARNING: HARDCODED VAUES
    train_dataset = RakutenImageDataset(
        image_folder,
        train_image_info["image_name"].to_list(),
        train_image_info["label"].to_list(),
        transform,
    )
    val_dataset = RakutenImageDataset(
        image_folder,
        val_image_info["image_name"].to_list(),
        val_image_info["label"].to_list(),
        transform,
    )
    test_dataset = RakutenImageDataset(
        image_folder,
        test_image_info["image_name"].to_list(),
        test_image_info["label"].to_list(),
        transform,
    )

    # Creating dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
