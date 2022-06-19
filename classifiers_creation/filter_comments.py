import csv
import gensim
import pandas as pd
from tqdm import tqdm
from dataset import TRAITS
from collections import Counter
from nltk.corpus import stopwords

PATH_DATASET = "/home/rcala/PromptMBTI_Masters/datasets/all_comments_with_mbti.csv"
DESTINATION_PATH = (
    "/home/rcala/PromptMBTI_Masters/datasets/filtered_comments_with_mbti.csv"
)

MIN_LENGTH = 5
MAX_LENGTH = 256

reddit_generic_sequences = [
    "https://",
    "http://",
    "Thank you for your submission!",
    "This comment or submission has been removed",
    "All replies to this post must be a maximum",
    "Here you can write whatever!",
    "^^^^^^^",
]

eng_stopwords_dict = Counter(stopwords.words("english"))
all_comments_with_mbti = pd.read_csv(PATH_DATASET, usecols=["body"] + TRAITS)
texts = list(all_comments_with_mbti["body"])
introverted = list(all_comments_with_mbti[TRAITS[0]])
intuitive = list(all_comments_with_mbti[TRAITS[1]])
thinking = list(all_comments_with_mbti[TRAITS[2]])
perceiving = list(all_comments_with_mbti[TRAITS[3]])
items = zip(texts, introverted, intuitive, thinking, perceiving)

filtered_items = list(items)

# length
filtered_from_length_items = []
for item in tqdm(filtered_items):
    tokenized = list(gensim.utils.tokenize(item[0]))
    if len(tokenized) > MIN_LENGTH and len(tokenized) < MAX_LENGTH:
        filtered_from_length_items += [item]

filtered_items = filtered_from_length_items

# generic
filtered_from_generic_items = []
for item in tqdm(filtered_items):

    contains = False
    for sequence in reddit_generic_sequences:
        if sequence in item[0]:
            contains = True
            break

    if contains == False:
        filtered_from_generic_items += [item]

filtered_items = filtered_from_generic_items

# natural
no_foreign = []
for item in tqdm(filtered_items):

    contains = False
    tokenized = list(gensim.utils.tokenize(item[0].lower()))

    for word in tokenized:
        if word in eng_stopwords_dict:
            contains = True
            break

    if contains == True:
        no_foreign += [item]

filtered_items = no_foreign

with open(DESTINATION_PATH, "w") as f:
    writer = csv.writer(f)
    f.write("text," + ",".join(TRAITS) + "\n")
    writer.writerows(filtered_items)
