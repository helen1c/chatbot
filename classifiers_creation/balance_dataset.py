import pandas as pd
import csv
from dataset import TRAITS

DATASET_PATH = "/home/rcala/PromptMBTI_Masters/datasets/filtered_comments_with_mbti.csv"
DESTINATION_PATH = "/home/rcala/PromptMBTI_Masters/datasets/filtered_comments_with_"
NUM_EXAMPLES = 1500000

dataset = pd.read_csv(DATASET_PATH)
dataset = dataset.sample(frac=1)
texts = list(dataset["text"])
labels = list(
    zip(
        list(dataset[TRAITS[0]]),
        list(dataset[TRAITS[1]]),
        list(dataset[TRAITS[2]]),
        list(dataset[TRAITS[3]]),
    )
)

for trait in range(len(TRAITS)):
    zero_trait = 0
    one_trait = 0
    new_texts = []
    new_labels = []
    for idx, text in enumerate(texts):
        if labels[idx][trait] == 0 and zero_trait < NUM_EXAMPLES:
            zero_trait += 1
            new_texts += [text]
            new_labels += [labels[idx][trait]]
        elif labels[idx][trait] == 1 and one_trait < NUM_EXAMPLES:
            one_trait += 1
            new_texts += [text]
            new_labels += [labels[idx][trait]]

    new_dataset = zip(new_texts, new_labels)

    with open(DESTINATION_PATH + TRAITS[trait] + ".csv", "w") as f:
        writer = csv.writer(f)
        f.write("text," + TRAITS[trait] + "\n")
        writer.writerows(new_dataset)
