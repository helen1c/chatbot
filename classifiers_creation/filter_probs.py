import csv
import pandas as pd
from dataset import TRAITS

CURR_TRAIT = 3
UNFILTERED_PATH = (
    "/home/rcala/PromptMBTI_Masters/filtered/"
    + "bert_probs_"
    + TRAITS[CURR_TRAIT]
    + ".csv"
)
FILTERED_PATH = (
    "/home/rcala/PromptMBTI_Masters/filtered/"
    + "bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + ".csv"
)
zero_label_threshold = [0.251, 0.055, 0.236, 0.314]
one_label_threshold = [0.726, 0.848, 0.763, 0.680]

unfiltered_items = pd.read_csv(UNFILTERED_PATH)
texts = list(unfiltered_items["text"])
probs = list(unfiltered_items[TRAITS[CURR_TRAIT]])

filtered_items = [
    item
    for item in zip(texts, probs)
    if item[1] < zero_label_threshold[CURR_TRAIT]
    or item[1] > one_label_threshold[CURR_TRAIT]
]

zero_labels = 0
one_labels = 0
total_examples = len(filtered_items)

for item in filtered_items:
    if item[1] > one_label_threshold[CURR_TRAIT]:
        one_labels += 1
    elif item[1] < zero_label_threshold[CURR_TRAIT]:
        zero_labels += 1

print("Zero labels: " + str(zero_labels))
print("One labels: " + str(one_labels))
print("All examples: " + str(total_examples))

with open(FILTERED_PATH, "w") as f:
    writer = csv.writer(f)
    f.write("text," + TRAITS[CURR_TRAIT] + "\n")
    writer.writerows(filtered_items)
