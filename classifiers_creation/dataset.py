import torch
import pandas as pd
from torch.utils.data import DataLoader

TRAITS = ["introverted", "intuitive", "thinking", "perceiving"]
OPPOSITE_TRAITS = ["extroverted", "sensing", "feeling", "judging"]


class MBTITraitDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        item = {"text": self.texts[idx], "label": self.labels[idx]}

        return item


def collate_classic_mbti(samples, tokenizer):

    texts = [sample["text"] for sample in samples]
    labels = [sample["label"] for sample in samples]

    inputs = tokenizer(
        text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
    )

    inputs.update({"labels": torch.tensor(labels)})

    return texts, inputs


def prepare_filter_splits(path, idx_split, trait, batch_size, tokenizer):

    all_comments_with_mbti = pd.read_csv(path, usecols=["text", TRAITS[trait]])

    all_comments_with_mbti = all_comments_with_mbti.sample(frac=1)

    texts = list(all_comments_with_mbti["text"])

    labels = list(all_comments_with_mbti[TRAITS[trait]])

    collate_fun = lambda samples: collate_classic_mbti(samples, tokenizer=tokenizer)

    if idx_split == 1:

        training_texts = texts[: int(2 / 3 * len(texts))]
        test_texts = texts[int(2 / 3 * len(texts)) :]
        train_texts = training_texts[: int(0.8 * len(training_texts))]
        val_texts = training_texts[int(0.8 * len(training_texts)) :]

        training_labels = labels[: int(2 / 3 * len(labels))]
        test_labels = labels[int(2 / 3 * len(labels)) :]
        train_labels = training_labels[: int(0.8 * len(training_labels))]
        val_labels = training_labels[int(0.8 * len(training_labels)) :]

    if idx_split == 2:

        training_texts = (
            texts[: int(1 / 3 * len(texts))] + texts[int(2 / 3 * len(texts)) :]
        )
        test_texts = texts[int(1 / 3 * len(texts)) : int(2 / 3 * len(texts))]
        train_texts = training_texts[: int(0.8 * len(training_texts))]
        val_texts = training_texts[int(0.8 * len(training_texts)) :]

        training_labels = (
            labels[: int(1 / 3 * len(labels))] + labels[int(2 / 3 * len(labels)) :]
        )
        test_labels = labels[int(1 / 3 * len(labels)) : int(2 / 3 * len(labels))]
        train_labels = training_labels[: int(0.8 * len(training_labels))]
        val_labels = training_labels[int(0.8 * len(training_labels)) :]

    if idx_split == 3:

        training_texts = texts[int(1 / 3 * len(texts)) :]
        test_texts = texts[: int(1 / 3 * len(texts))]
        train_texts = training_texts[: int(0.8 * len(training_texts))]
        val_texts = training_texts[int(0.8 * len(training_texts)) :]

        training_labels = labels[int(1 / 3 * len(labels)) :]
        test_labels = labels[: int(1 / 3 * len(labels))]
        train_labels = training_labels[: int(0.8 * len(training_labels))]
        val_labels = training_labels[int(0.8 * len(training_labels)) :]

    train_dataset = MBTITraitDataset(train_texts, train_labels)
    val_dataset = MBTITraitDataset(val_texts, val_labels)
    test_dataset = MBTITraitDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fun,)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate_fun,)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fun,)

    return train_loader, val_loader, test_loader


def prepare_mbti_splits(path, batch_size, tokenizer):

    filtered_texts_with_mbti = pd.read_csv(path)

    filtered_texts_with_mbti = filtered_texts_with_mbti.sample(frac=1)

    texts = list(filtered_texts_with_mbti["text"])
    trait_column = filtered_texts_with_mbti.columns[1]
    labels = list(filtered_texts_with_mbti[trait_column])

    collate_fun = lambda samples: collate_classic_mbti(samples, tokenizer=tokenizer)

    train_texts = texts[: int(0.7 * len(texts))]
    val_texts = texts[int(0.7 * len(texts)) : int(0.8 * len(texts))]
    test_texts = texts[int(0.8 * len(texts)) :]

    train_labels = labels[: int(0.7 * len(labels))]
    val_labels = labels[int(0.7 * len(labels)) : int(0.8 * len(labels))]
    test_labels = labels[int(0.8 * len(labels)) :]

    train_dataset = MBTITraitDataset(train_texts, train_labels)
    val_dataset = MBTITraitDataset(val_texts, val_labels)
    test_dataset = MBTITraitDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fun,)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate_fun,)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fun,)

    return train_loader, val_loader, test_loader

