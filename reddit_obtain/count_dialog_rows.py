import os

DIALOGS_PATH = "/home/rcala/chatbot/reddit_obtain/Reddit/dialogs"

dataset_paths = []
for (dirpath, dirnames, filenames) in os.walk(DIALOGS_PATH):
    dataset_paths += [os.path.join(dirpath, file) for file in filenames]

num_dialogs = 0

for dataset_path in dataset_paths:
    num_dialogs += len(open(dataset_path, "r", errors="ignore").readlines())

print("Num dialogs: {}".format(num_dialogs))
