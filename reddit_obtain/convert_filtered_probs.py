import csv

#convert for dialogpt

FILTERED_PATH_CSV = (
    "/mnt/rcala/mbti_probs/preprocessed_reddit_dialogs_introverted_filtered_0.9.csv"
)

FILTERED_PATH_TSV = (
    "/mnt/rcala/filtered_files/preprocessed_reddit_dialogs_introverted_filtered_0.9.tsv"
)

with open(FILTERED_PATH_CSV,"r") as filtered_csv_file:

    filtered_csv_file.readline()

    filtered_csv_reader = csv.reader(filtered_csv_file)

    with open(FILTERED_PATH_TSV,"w") as filtered_tsv_file:

        for row in filtered_csv_reader:

            filtered_tsv_file.write(row[0]+"\n")