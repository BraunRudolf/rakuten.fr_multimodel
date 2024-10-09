import torch
from torch.utils.data import DataLoader

from src.dataset.dataset import (
    RakutenFusionDataset,
    RakutenImageDataset,
    RakutenTextDataset,
    RakutenTextTransformerDataset,
)
from src.dataset.preprocess import (
    get_oversampling_sampler,
    get_undersampling_sampler,
    retrieve_image_info,
)


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
    preprocessing_pipeline=[],
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
        preprocessing_pipeline,
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
        preprocessing_pipeline,
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
        preprocessing_pipeline,
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
    sampling_mode=None,
):
    """
    Function to create a set of train/val/test dataloaders
    """

    # Creating dataloaders

    # Check if sampling mode is specified
    if sampling_mode:
        # Retrieve the labels for the dataset
        targets = torch.tensor(
            [sample[1] for sample in train_dataset]
        )  # Assuming train_dataset returns (data, label)

        # Choose the appropriate sampler based on the mode
        if sampling_mode == "oversample":
            sampler = get_oversampling_sampler(train_dataset, targets)
        elif sampling_mode == "undersample":
            sampler = get_undersampling_sampler(train_dataset, targets)
        else:
            raise ValueError(
                f"Invalid sampling_mode: {sampling_mode}. Use 'oversample' or 'undersample'."
            )

        # Set the DataLoader to use the sampler instead of shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Use the sampler here
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    else:
        # Default behavior with shuffling
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


def create_text_transformer_datasets(
    db_url,
    table_name,
    mapping_table_name,
    text_column,
    label_column,
    mapping_column,
    # vocab,
    tokenizer,
    train_indices,
    val_indices,
    test_indices,
    preprocessing_pipeline=[],
):
    train_dataset = RakutenTextTransformerDataset(
        db_url,
        table_name,
        mapping_table_name,
        text_column,
        label_column,
        mapping_column,
        # vocab,
        train_indices,
        tokenizer,
        preprocessing_pipeline,
    )
    val_dataset = RakutenTextTransformerDataset(
        db_url,
        table_name,
        mapping_table_name,
        text_column,
        label_column,
        mapping_column,
        # vocab,
        val_indices,
        tokenizer,
        preprocessing_pipeline,
    )
    test_dataset = RakutenTextTransformerDataset(
        db_url,
        table_name,
        mapping_table_name,
        text_column,
        label_column,
        mapping_column,
        # vocab,
        test_indices,
        tokenizer,
        preprocessing_pipeline,
    )
    return train_dataset, val_dataset, test_dataset
