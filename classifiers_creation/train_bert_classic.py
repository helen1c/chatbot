import os
import torch
import sys
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import TRAITS
from dataset import prepare_classic_mbti_splits
from pytorchtools import EarlyStopping

CURR_TRAIT = 3

PATH_DATASET = (
    "/home/rcala/PromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + "_discrete"
    + ".csv"
)
BERT_MODEL_PATH = "bert-base-uncased"


BERT_SAVE_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "bert"
        + "_"
        + TRAITS[CURR_TRAIT]
        + "_classic"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

random_seed = 1

torch.manual_seed(random_seed)
set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

model_config = BertConfig.from_pretrained(
    pretrained_model_name_or_path=BERT_MODEL_PATH, num_labels=2
)
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_PATH, config=model_config
)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, do_lower_case=True)

model.to(dev)

earlystopping = EarlyStopping(patience=5, path=BERT_SAVE_PATH)
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 100
batch_size = 2

train_loader, val_loader, test_loader = prepare_classic_mbti_splits(
    PATH_DATASET, batch_size, tokenizer
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

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        tqdm_train.set_description(
            "Epoch {}, Train batch_loss: {}".format(epoch + 1, loss.item(),)
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

            loss = loss.mean()

            all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
            all_true += list(inputs["labels"].cpu().detach().numpy())

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
