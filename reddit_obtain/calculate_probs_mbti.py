import os
import torch
import csv
import torch.nn.functional as F
import numpy as np
import random
from reddit_mbti import prepare_mbti_dialogs
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)
from reddit_mbti import TRAITS
from tqdm import tqdm
import time

CURR_TRAIT = 3
PATH_DATASET = (
    "/mnt/rcala/dialog_files/first_iteration/preprocessed_reddit_dialogs.tsv"
)
PROBS_PATH = (
    "/mnt/rcala/mbti_probs/first_iteration/preprocessed_reddit_dialogs_"+TRAITS[CURR_TRAIT]+"_probs.csv"
)
BERT_LOAD_PATH = (
    "/mnt/rcala/filter_models/" + "bert" + "_" + TRAITS[CURR_TRAIT] + "_classic"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

model_config = BertConfig.from_pretrained(
    pretrained_model_name_or_path=BERT_LOAD_PATH, num_labels=2
)
model = BertForSequenceClassification.from_pretrained(BERT_LOAD_PATH, config=model_config)
tokenizer = BertTokenizer.from_pretrained(BERT_LOAD_PATH, do_lower_case=True)

model.to(dev)

batch_size = 16

dataloader = prepare_mbti_dialogs(
    PATH_DATASET, batch_size, tokenizer
)

model.eval()

tqdm_loader = tqdm(dataloader)

time.sleep(10)

with torch.no_grad():

    with open(PROBS_PATH, "w") as f:

        f.write("dialog," + TRAITS[CURR_TRAIT] + "\n")
        writer = csv.writer(f)

        for dialogs, bot_texts, inputs in tqdm_loader:

            inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

            prediction = model(**inputs)[:1][0]

            probs = F.softmax(prediction, dim=1)[:, 1].tolist()

            writer.writerows(zip(dialogs,probs))

