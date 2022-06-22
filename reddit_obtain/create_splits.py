import random

TRAIN_SIZE = 500000
EVAL_SIZE = 50000
FILTERED_PATH_TSV = (
    "/mnt/rcala/filtered_files/first_iteration_lang/preprocessed_reddit_dialogs_judging_filtered_0.122.tsv"
)

random.seed(123)

with open(FILTERED_PATH_TSV,"r") as filtered_file:

    lines = filtered_file.readlines()
    random.shuffle(lines)
    lines = lines[:TRAIN_SIZE+EVAL_SIZE]

    with open(FILTERED_PATH_TSV[:-4]+"_train.tsv","w") as train_file:

        for line in lines[:TRAIN_SIZE]:
            train_file.write(line)

    with open(FILTERED_PATH_TSV[:-4]+"_eval.tsv","w") as eval_file:

        for line in lines[TRAIN_SIZE:TRAIN_SIZE+EVAL_SIZE]:
            eval_file.write(line)



