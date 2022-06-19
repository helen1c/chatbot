import csv
from reddit_mbti import TRAITS, OPPOSITE_TRAITS
from tqdm import tqdm

CURR_TRAIT = 3
OPPOSITE_TRAIT = True
WRITE = True

PROBS_PATH = (
    "/mnt/rcala/mbti_probs/first_iteration_lang/preprocessed_reddit_dialogs_"
    + TRAITS[CURR_TRAIT]
    + "_probs.csv"
)

opposite_trait_threshold = [0.204, 0.017, 0.205, 0.122]
trait_threshold = [0.815, 0.891, 0.818, 0.893]


if OPPOSITE_TRAIT:
    FILTERED_PATH = (
        "/mnt/rcala/mbti_probs/first_iteration_lang/preprocessed_reddit_dialogs_"
        + OPPOSITE_TRAITS[CURR_TRAIT]
        + "_filtered_"
        + str(opposite_trait_threshold[CURR_TRAIT])
        + ".csv"
    )
else:
    FILTERED_PATH = (
        "/mnt/rcala/mbti_probs/first_iteration_lang/preprocessed_reddit_dialogs_"
        + TRAITS[CURR_TRAIT]
        + "_filtered_"
        + str(trait_threshold[CURR_TRAIT])
        + ".csv"
    )

with open(PROBS_PATH, "r") as probs_file:

    probs_file.readline()

    csv_probs = csv.reader(line.replace("\0", "") for line in probs_file)
    num_lines = 0

    if WRITE:
        filtered_file = open(FILTERED_PATH, "w")
        filtered_file.write("dialog," + TRAITS[CURR_TRAIT] + "\n")
        csv_filtered = csv.writer(filtered_file)

    num_filtered_lines = 0

    for row in tqdm(csv_probs):

        num_lines += 1

        if OPPOSITE_TRAIT:
            if float(row[1]) < opposite_trait_threshold[CURR_TRAIT]:
                num_filtered_lines += 1
                if WRITE:
                    csv_filtered.writerow(row)
        else:
            if float(row[1]) > trait_threshold[CURR_TRAIT]:
                num_filtered_lines += 1
                if WRITE:
                    csv_filtered.writerow(row)

if WRITE:
    filtered_file.close()


print("All examples: " + str(num_lines))
print("Left examples: " + str(num_filtered_lines))
