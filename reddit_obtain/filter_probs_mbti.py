import csv
from reddit_mbti import TRAITS, OPPOSITE_TRAITS
from tqdm import tqdm

CURR_TRAIT = 0
OPPOSITE_TRAIT = False

PROBS_PATH = (
    "/mnt/rcala/mbti_probs/preprocessed_reddit_dialogs_"+TRAITS[CURR_TRAIT]+"_probs.csv"
)

opposite_trait_threshold = [0.1, 0.142, 0.229, 0.318]
trait_threshold = [0.9, 0.947, 0.769, 0.767]


if OPPOSITE_TRAIT:
    FILTERED_PATH= (
        "/mnt/rcala/mbti_probs/preprocessed_reddit_dialogs_"+OPPOSITE_TRAITS[CURR_TRAIT]+"_filtered_"+str(opposite_trait_threshold[CURR_TRAIT])+".csv"
    )
else:
    FILTERED_PATH= (
        "/mnt/rcala/mbti_probs/preprocessed_reddit_dialogs_"+TRAITS[CURR_TRAIT]+"_filtered_"+str(trait_threshold[CURR_TRAIT])+".csv"
    ) 

with open(PROBS_PATH,"r") as probs_file:

    probs_file.readline()

    csv_probs = csv.reader(probs_file)
    num_lines = 0

    with open(FILTERED_PATH,"w") as filtered_file:

        filtered_file.write("dialog,"+TRAITS[CURR_TRAIT]+"\n")

        csv_filtered = csv.writer(filtered_file)
        num_filtered_lines = 0

        for row in tqdm(csv_probs):

            num_lines += 1

            if OPPOSITE_TRAIT:
                if float(row[1]) < opposite_trait_threshold[CURR_TRAIT]:
                    num_filtered_lines += 1
                    csv_filtered.writerow(row)
            else:
                if float(row[1]) > trait_threshold[CURR_TRAIT]:
                    num_filtered_lines += 1
                    csv_filtered.writerow(row)
            
print("All examples: " + str(num_lines))
print("Left examples: " + str(num_filtered_lines))
