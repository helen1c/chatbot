import os
from tqdm import tqdm

DIALOGS_PATH = "/home/rcala/chatbot/reddit_obtain/Reddit/dialogs"

dataset_paths = []
for (dirpath, dirnames, filenames) in os.walk(DIALOGS_PATH):
    dataset_paths += [os.path.join(dirpath, file) for file in filenames]

num_dialogs = 0
num_dialogs_file = {}

print("Per file:")
for dataset_path in dataset_paths:
    with open(dataset_path, "r", errors="ignore") as read_file:
        num_read_file = len(read_file.readlines())
        num_dialogs += num_read_file
        num_dialogs_file[dataset_path] = num_read_file
        print(dataset_path+" "+str(num_dialogs_file[dataset_path]))


print("Num dialogs: {}".format(num_dialogs))
