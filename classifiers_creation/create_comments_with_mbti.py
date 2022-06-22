import pandas as pd
import math
from tqdm import tqdm

all_comments = pd.read_csv("/home/rcala/pandora/all_comments_since_2015.csv")
profiles = pd.read_csv("/home/rcala/pandora/author_profiles.csv")

introverted = {}
intuitive = {}
thinking = {}
perceiving = {}

total_mbti_none = 0
total_mbti = 0

print("Total initial mbti: " + str(len(profiles)))

for idx, line in profiles.iterrows():

    if math.isnan(line["introverted"]):
        total_mbti_none += 1
        continue
    if math.isnan(line["intuitive"]):
        total_mbti_none += 1
        continue
    if math.isnan(line["thinking"]):
        total_mbti_none += 1
        continue
    if math.isnan(line["perceiving"]):
        total_mbti_none += 1
        continue

    introverted[line["author"]] = int(line["introverted"])
    intuitive[line["author"]] = int(line["intuitive"])
    thinking[line["author"]] = int(line["thinking"])
    perceiving[line["author"]] = int(line["perceiving"])

    total_mbti += 1

print("Total mbti: " + str(total_mbti))
print("Total mbti none: " + str(total_mbti_none))

all_comments["introverted"] = [1] * len(all_comments)
all_comments["intuitive"] = [1] * len(all_comments)
all_comments["thinking"] = [1] * len(all_comments)
all_comments["perceiving"] = [1] * len(all_comments)

print("Total initial comments: " + str(len(all_comments)))

for idx, line in tqdm(all_comments.iterrows()):

    all_comments.at[idx, "introverted"] = introverted.get(line["author"])
    all_comments.at[idx, "intuitive"] = intuitive.get(line["author"])
    all_comments.at[idx, "thinking"] = thinking.get(line["author"])
    all_comments.at[idx, "perceiving"] = perceiving.get(line["author"])

all_comments = all_comments[all_comments["introverted"].notnull()]
all_comments = all_comments[all_comments["intuitive"].notnull()]
all_comments = all_comments[all_comments["thinking"].notnull()]
all_comments = all_comments[all_comments["perceiving"].notnull()]

print("Total comments after: " + str(len(all_comments)))

all_comments.to_csv(
    "/home/rcala/PromptMBTI_Masters/datasets/all_comments_with_mbti.csv"
)
