import csv
import pandas as pd
from reddit_mbti import TRAITS, OPPOSITE_TRAITS

CURR_TRAIT = 0
OPPOSITE_TRAIT = False
PROBS_PATH = (
    "reddit_obtain/Reddit/mbti_probs/reddit_dialogs_"+TRAITS[CURR_TRAIT]+"_probs.csv"
)
if OPPOSITE_TRAIT:
    FILTERED_PATH= (
        "reddit_obtain/Reddit/mbti_filtered/reddit_dialogs_"+OPPOSITE_TRAITS[CURR_TRAIT]+"_filtered.tsv"
    )
else:
    FILTERED_PATH= (
        "reddit_obtain/Reddit/mbti_filtered/reddit_dialogs_"+TRAITS[CURR_TRAIT]+"_filtered.tsv"
    ) 

opposite_trait_threshold = [0.217, 0.142, 0.229, 0.318]
trait_threshold = [0.736, 0.947, 0.769, 0.767]

probs_items = pd.read_csv(PROBS_PATH)
dialogs = list(probs_items["dialog"])
probs = list(probs_items[TRAITS[CURR_TRAIT]])

if OPPOSITE_TRAIT:
    filtered_items = [item for item in zip(dialogs, probs) if item[1] < opposite_trait_threshold[CURR_TRAIT]]
else:
    filtered_items = [item for item in zip(dialogs, probs) if item[1] > trait_threshold[CURR_TRAIT]]

print("All examples: " + str(len(dialogs)))
print("Left examples: " + str(len(filtered_items)))

with open(FILTERED_PATH, "w") as f:
    for item in filtered_items:
        f.write(item[0]+"\n")
