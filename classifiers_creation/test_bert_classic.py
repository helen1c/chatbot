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
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import prepare_classic_mbti_splits
from dataset import TRAITS

CURR_TRAIT = 3

PATH_DATASET = (
    "/home/rcala/PromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + "_discrete"
    + ".csv"
)

BERT_LOAD_PATH = (
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
    pretrained_model_name_or_path=BERT_LOAD_PATH, num_labels=2
)
model = BertForSequenceClassification.from_pretrained(
    BERT_LOAD_PATH, config=model_config
)
tokenizer = BertTokenizer.from_pretrained(BERT_LOAD_PATH, do_lower_case=True)

model.to(dev)

batch_size = 16

train_loader, val_loader, test_loader = prepare_classic_mbti_splits(
    PATH_DATASET, batch_size, tokenizer
)

model.eval()
all_pred = []
all_true = []

tqdm_test = tqdm(test_loader)

with torch.no_grad():
    for texts, inputs in tqdm_test:

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        loss, prediction = model(**inputs)[:2]

        loss = loss.mean()
        all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
        all_true += list(inputs["labels"].cpu().detach().numpy())

        tqdm_test.set_description("Test batch_loss: {}".format(loss.item()))

test_acc = accuracy_score(all_true, all_pred)
test_f1 = f1_score(all_true, all_pred, average="macro")

print(f"Test_acc: {test_acc:.4f} Test_f1: {test_f1:.4f}")
