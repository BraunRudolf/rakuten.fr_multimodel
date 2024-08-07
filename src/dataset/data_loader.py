from torch.utils.data import DataLoader

from src.dataset.dataset import RakutenFusionDataset, RakutenImageDataset, RakutenTextDataset
from src.dataset.preprocess import retrieve_image_info


def create_text_datasets(
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
):
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
    return train_dataset, val_dataset, test_dataset


def create_image_datasets(
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
    image_folder="images",
    transform=None,
):
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

    return train_dataset, val_dataset, test_dataset


def create_fusion_datasets(
    train_text_dataset,
    val_text_dataset,
    test_text_dataset,
    train_image_dataset,
    val_image_dataset,
    test_image_dataset,
):
    train_fusion_dataset = RakutenFusionDataset(train_text_dataset, train_image_dataset)
    val_fusion_dataset = RakutenFusionDataset(val_text_dataset, val_image_dataset)
    test_fusion_dataset = RakutenFusionDataset(test_text_dataset, test_image_dataset)

    return train_fusion_dataset, val_fusion_dataset, test_fusion_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_workers,
    pin_memory,
    shuffle,
    collate_fn=None,
):
    """
    Function to create a set of train/val/test dataloaders
    """

    # Creating dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
