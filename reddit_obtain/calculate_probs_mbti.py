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
    set_seed,
)
from reddit_mbti import TRAITS
from tqdm import tqdm

CURR_TRAIT = 0
PATH_DATASET = (
    "reddit_obtain/Reddit/reddit_dialogs.tsv"
)
PROBS_PATH = (
    "reddit_obtain/Reddit/mbti_probs/reddit_dialogs_"+TRAITS[CURR_TRAIT]+"_probs.tsv"
)
BERT_LOAD_PATH = (
    "/home/rcala/PromptMBTI_Masters/models/" + "bert" + "_" + TRAITS[CURR_TRAIT] + "_classic"
)
BERT_LOAD_PATH = (
    "/home/rcala/PromptMBTI_Masters/models/" + "General_TinyBERT_4L_312D/"
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
    pretrained_model_name_or_path=BERT_LOAD_PATH, num_labels=2
)
model = BertForSequenceClassification.from_pretrained(BERT_LOAD_PATH, config=model_config)
tokenizer = BertTokenizer.from_pretrained(BERT_LOAD_PATH, do_lower_case=True)

model.to(dev)

batch_size = 16

dialogs, dataloader = prepare_mbti_dialogs(
    PATH_DATASET, batch_size, tokenizer
)

model.eval()


total_probs = []
tqdm_loader = tqdm(dataloader)

with torch.no_grad():
    for texts, inputs in tqdm_loader:

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        prediction = model(**inputs)[:1][0]

        total_probs += F.softmax(prediction, dim=1)[:, 1].tolist()

final_items = zip(dialogs, total_probs)

with open(PROBS_PATH, "w") as f:
    writer = csv.writer(f)
    f.write("dialog," + TRAITS[CURR_TRAIT] + "\n")
    writer.writerows(final_items)

