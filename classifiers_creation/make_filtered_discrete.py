import pandas as pd
from tqdm import tqdm
from dataset import TRAITS

CURR_TRAIT = 3

PROMPT_LABELS = ["introverted", "intuitive", "thinking", "perceiving"]
PROMPT_OPPOSITE_LABELS = ["extroverted", "sensing", "feeling", "judging"]

path = (
    "/home/rcala/PromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + ".csv"
)
original_path = (
    "/home/rcala/PromptMBTI_Masters/datasets/filtered_comments_with_"
    + TRAITS[CURR_TRAIT]
    + ".csv"
)
dataset = pd.read_csv(path)
original_dataset = pd.read_csv(original_path)
discrete_dataset = dataset.copy(deep=True)
prompt_dataset = dataset.copy(deep=True)
discrete = []
prompt = []

labels = {}
for idx, line in tqdm(original_dataset.iterrows()):
    labels[line["text"]] = line[TRAITS[CURR_TRAIT]]

for idx, line in tqdm(dataset.iterrows()):
    discrete += [labels.get(line["text"])]

discrete_dataset[TRAITS[CURR_TRAIT]] = discrete
discrete_dataset.to_csv(path[:-4] + "_discrete" + ".csv", index=False)

for idx, line in tqdm(discrete_dataset.iterrows()):
    prompt += [
        PROMPT_LABELS[CURR_TRAIT]
        if line[TRAITS[CURR_TRAIT]] == 1
        else PROMPT_OPPOSITE_LABELS[CURR_TRAIT]
    ]

prompt_dataset = prompt_dataset.drop(TRAITS[CURR_TRAIT], 1)
prompt_dataset["label"] = prompt
prompt_dataset.to_csv(path[:-4] + "_prompt" + ".csv", index=False)
