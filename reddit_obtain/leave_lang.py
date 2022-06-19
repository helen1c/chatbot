from argparse import ArgumentParser
from tqdm import tqdm
import csv

parser = ArgumentParser()
parser.add_argument("--input_path", default="/mnt/rcala/mbti_probs/first_iteration/preprocessed_reddit_dialogs_perceiving_probs.csv")
parser.add_argument("--ref_lang_path",default="/mnt/rcala/dialog_files/first_iteration/preprocessed_reddit_dialogs_lang.tsv")
parser.add_argument("--output_path", default="/mnt/rcala/mbti_probs/first_iteration_lang/preprocessed_reddit_dialogs_perceiving_probs.csv")
args = parser.parse_args()

left_lang = {}

with open(args.ref_lang_path,"r") as f: 
    for line in tqdm(f):
        left_lang[line[:-1]]=1

with open(args.input_path,"r") as not_filtered_file:
    with open(args.output_path,"w") as filtered_file:
        filtered_file.write(next(not_filtered_file))

        not_filtered_csv_reader = csv.reader(line.replace('\0', '') for line in not_filtered_file)
        filtered_csv_writer = csv.writer(filtered_file)

        for row in tqdm(not_filtered_csv_reader):
            if row[0] in left_lang:
                filtered_csv_writer.writerow(row)