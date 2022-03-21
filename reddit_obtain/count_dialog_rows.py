import os

DIALOGS_PATH = "/home/rcala/chatbot/reddit_obtain/Reddit/dialogs"

dataset_paths = []
for (dirpath, dirnames, filenames) in os.walk(DIALOGS_PATH):
    dataset_paths += [os.path.join(dirpath, file) for file in filenames]

num_dialogs = 0
num_dialogs_file = {}

for dataset_path in dataset_paths:
    num_dialogs += len(open(dataset_path, "r", errors="ignore").readlines())
    num_dialogs_file[dataset_path]=len(open(dataset_path, "r", errors="ignore").readlines())


print("Num dialogs: {}".format(num_dialogs))
print("Per file:")
for file in num_dialogs_file:
    print(file+" "+str(num_dialogs_file[file]))
