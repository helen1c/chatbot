import csv

#convert for dialogpt

FILTERED_PATH_CSV = (
    "/mnt/rcala/mbti_probs/first_iteration_lang/preprocessed_reddit_dialogs_judging_filtered_0.122.csv"
)

FILTERED_PATH_TSV = (
    "/mnt/rcala/filtered_files/first_iteration_lang/preprocessed_reddit_dialogs_judging_filtered_0.122.tsv"
)

with open(FILTERED_PATH_CSV,"r") as filtered_csv_file:

    filtered_csv_file.readline()

    filtered_csv_reader = csv.reader(filtered_csv_file)

    with open(FILTERED_PATH_TSV,"w") as filtered_tsv_file:

        for row in filtered_csv_reader:

            filtered_tsv_file.write(row[0]+"\n")