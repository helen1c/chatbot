import os
import torch
import numpy as np
import random
import torch.nn.functional as F
import csv
import sys
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import DataParallel
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    set_seed,
)
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import prepare_filter_splits
from dataset import TRAITS

CURR_TRAIT = 3
IDX_SPLIT = 3
BERT_LOAD_PATH = (
    "/home/rcala/PromptMBTI_Masters/models/"
    + "bert"
    + "_"
    + TRAITS[CURR_TRAIT]
    + "_"
    + str(IDX_SPLIT)
)
FILTERED_PATH = (
    "/home/rcala/PromptMBTI_Masters/filtered/"
    + "bert_probs"
    + "_"
    + TRAITS[CURR_TRAIT]
    + "_"
    + str(IDX_SPLIT)
    + ".csv"
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

random_seed = 123

torch.manual_seed(random_seed)
set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

model_config = BertConfig.from_pretrained(
    pretrained_model_name_or_path=BERT_LOAD_PATH, num_labels=2
)
model = BertForSequenceClassification.from_pretrained(
    BERT_LOAD_PATH, config=model_config
)
tokenizer = BertTokenizer.from_pretrained(BERT_LOAD_PATH, do_lower_case=True)

model = model.to(dev)

batch_size = 32

train_loader, val_loader, test_loader = prepare_filter_splits(
    PATH_DATASET,
    idx_split=IDX_SPLIT,
    trait=CURR_TRAIT,
    batch_size=batch_size,
    tokenizer=tokenizer,
)

model.eval()

tqdm_test = tqdm(test_loader)

total_texts = []
total_probs = []
all_pred = []
all_true = []

with torch.no_grad():
    for texts, inputs in tqdm_test:

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        loss, prediction = model(**inputs)[:2]

        loss = loss.mean()
        all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
        all_true += list(inputs["labels"].cpu().detach().numpy())

        tqdm_test.set_description("Test batch_loss: {}".format(loss.item()))

        total_texts += texts
        total_probs += F.softmax(prediction, dim=1)[:, 1].tolist()


test_acc = accuracy_score(all_true, all_pred)
test_f1 = f1_score(all_true, all_pred, average="macro")

print(
    f"Test_acc: {test_acc:.4f}\
    Test_f1: {test_f1:.4f}"
)

final_items = zip(total_texts, total_probs)

with open(FILTERED_PATH, "w") as f:
    writer = csv.writer(f)
    f.write("text," + TRAITS[CURR_TRAIT] + "\n")
    writer.writerows(final_items)
