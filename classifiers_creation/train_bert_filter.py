import os
import torch
import sys
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    AdamW,
    set_seed,
    get_linear_schedule_with_warmup,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import prepare_filter_splits
from dataset import TRAITS
from pytorchtools import EarlyStopping

BERT_TINY_PATH = "/home/rcala/PromptMBTI_Masters/models/General_TinyBERT_4L_312D/"
CURR_TRAIT = 3
IDX_SPLIT = 3
BERT_SAVE_PATH = (
    "/home/rcala/PromptMBTI_Masters/models/"
    + "bert"
    + "_"
    + TRAITS[CURR_TRAIT]
    + "_"
    + str(IDX_SPLIT)
)
PATH_DATASET = (
    "/home/rcala/PromptMBTI_Masters/datasets/filtered_comments_with_"
    + TRAITS[CURR_TRAIT]
    + ".csv"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

torch.manual_seed(123)
set_seed(123)
np.random.seed(123)
random.seed(123)

model_config = BertConfig.from_pretrained(
    pretrained_model_name_or_path=BERT_TINY_PATH, num_labels=2,
)
model = BertForSequenceClassification.from_pretrained(
    BERT_TINY_PATH, config=model_config
)
tokenizer = BertTokenizer.from_pretrained(BERT_TINY_PATH, do_lower_case=True)

model = model.to(dev)

earlystopping = EarlyStopping(patience=2, path=BERT_SAVE_PATH)
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 10
batch_size = 16

train_loader, val_loader, test_loader = prepare_filter_splits(
    PATH_DATASET,
    idx_split=IDX_SPLIT,
    trait=CURR_TRAIT,
    batch_size=batch_size,
    tokenizer=tokenizer,
)

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps,
)

for epoch in range(epochs):

    model.train()
    all_pred = []
    all_true = []

    tqdm_train = tqdm(train_loader)

    for texts, inputs in tqdm_train:

        optimizer.zero_grad()
        model.zero_grad()

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        loss, prediction = model(**inputs)[:2]

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
        all_true += list(inputs["labels"].cpu().detach().numpy())

        tqdm_train.set_description(
            "Epoch {}, Train batch_loss: {}".format(epoch + 1, loss.item())
        )

        scheduler.step()

    train_acc = accuracy_score(all_true, all_pred)
    train_f1 = f1_score(all_true, all_pred, average="macro")

    model.eval()
    all_pred = []
    all_true = []

    tqdm_val = tqdm(val_loader)

    with torch.no_grad():
        for texts, inputs in tqdm_val:

            inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

            loss, prediction = model(**inputs)[:2]

            all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
            all_true += list(inputs["labels"].cpu().detach().numpy())

            loss = loss.mean()

            tqdm_val.set_description(
                "Epoch {}, Val batch_loss: {}".format(epoch + 1, loss.item())
            )

    val_acc = accuracy_score(all_true, all_pred)
    val_f1 = f1_score(all_true, all_pred, average="macro")

    print(f"Epoch {epoch+1}")
    print(f"Train_acc: {train_acc:.4f} Train_f1: {train_f1:.4f}")
    print(f"Val_acc: {val_acc:.4f} Val_f1: {val_f1:.4f}")
    earlystopping(-val_f1, model, tokenizer)
    print()
    if earlystopping.early_stop == True:
        break
