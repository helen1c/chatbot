from argparse import ArgumentParser
import re, os
from tqdm import tqdm
import sys

# url_re = "https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
url_re = "https://|http://"
write_to_file = False

parser = ArgumentParser()
parser.add_argument("--data_path", default="data/train_reddit_full_raw.tsv")
parser.add_argument("--output_folder", default="data/outputs")
parser.add_argument("--preprocessed_file_name", default="preprocessed_reddit.tsv")
parser.add_argument("--blacklist_path", default="data/words_blacklist_base.txt")
args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

words_black_or = open(args.blacklist_path, "r").readlines()
words_black_or = [word.strip().lower() for word in words_black_or]
words_blacklist = ["[removed]", "[deleted]"]
words_blacklist.extend(words_black_or)
words_blacklist.remove("ass")

data = open(args.data_path, "r").readlines()


def filter_instance(instance, w_blacklist=[], debug=True):
    instance = instance.lower()

    for word in w_blacklist:
        if word in instance:
            if debug:
                print(
                    f"BLACKLIST WORD SKIP: {instance}\n\nFOUND WORD: {word}\n\n",
                    file=sys.stderr,
                )
            return True, 0
    if re.search("[\[\]\(\)]", instance) != None:
        if debug:
            print(
                f"MARKUP SKIP: {instance}\n\n",
                file=sys.stderr,
            )
        return True, 1

    if re.search(url_re, instance):
        if debug:
            print(
                f"URL SKIP: {instance}\n\n",
                file=sys.stderr,
            )
        return True, 3

    reps = False
    tkns = instance.split()
    for i in range(2, len(tkns)):
        if tkns[i - 2] == tkns[i] and tkns[i - 1] == tkns[i]:
            reps = True
            break
    if reps:
        if debug:
            print(f"REPETITION SKIP: {instance}\n\n", file=sys.stderr)
        return True, 2

    return (False, -1)


counter = 0

final_dialogs = []

translations = {0: "blacklist", 1: "markup", 2: "repetition", 3: "http"}

rules_num = 4
out = {}
for i in range(rules_num):
    out[i] = 0

for dialog in tqdm(data):
    b, i = filter_instance(dialog, words_blacklist, debug=False)
    if not b:
        instance = dialog
        instance = instance.replace(chr(92), "")
        instance = (
            instance.replace("b/c", "because")
            .replace("j/k", "just kidding")
            .replace("w/o", "without")
            .replace("w/", "with")
        )
        instance = instance.replace("EOS", "")
        final_dialogs.append(instance)
    else:
        out[i] += 1

if write_to_file:
    open(os.path.join(args.output_folder, args.preprocessed_file_name), "w").writelines(
        final_dialogs
    )
else:
    print(f"NUMBER OF DIALOGS BEFORE PREPROCESSING: {len(data)}")
    print(f"NUMBER OF DIALOGS AFTER PREPROCESSING: {len(final_dialogs)}")
    print(f"REMOVED DIALOGS: {len(data) - len(final_dialogs)}")
    print(f"REMOVAL PER CATEGORY:")
    for k in out.keys():
        print(f"Removed: {translations[k]}={out[k]}")
