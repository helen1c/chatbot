from dataset import TRAITS

CURR_TRAIT = 3

root = "/home/rcala/PromptMBTI_Masters/filtered/"
file_names = [
    "bert_probs_" + TRAITS[CURR_TRAIT] + "_" + str(i) + ".csv" for i in range(1, 4)
]
destination_file_name = "bert_probs_" + TRAITS[CURR_TRAIT] + ".csv"

header = open(root + file_names[0], "r").readline()

all_lines = []

for file in file_names:
    f = open(root + file, "r")
    lines = f.readlines()
    lines = lines[1:]
    all_lines += lines

destination_f = open(root + destination_file_name, "w")
destination_f.write(header)
destination_f.writelines(all_lines)
