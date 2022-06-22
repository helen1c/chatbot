from argparse import ArgumentParser
import re, os
from tqdm import tqdm
import sys
from collections import Counter
import gensim
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from langdetect import detect
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# url_re = "https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
url_re = "https://|http://"
write_to_file = True
eng_stopwords_dict = Counter(stopwords.words('english'))

def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

parser = ArgumentParser()
parser.add_argument("--data_path", default="/mnt/rcala/dialog_files/first_iteration/preprocessed_reddit_dialogs.tsv")
parser.add_argument("--output_folder", default="/mnt/rcala/dialog_files/first_iteration/")
parser.add_argument("--preprocessed_file_name", default="preprocessed_reddit_dialogs_lang1.tsv")
parser.add_argument("--blacklist_path", default="/home/rcala/chatbot/reddit_obtain/prepro_files/words_blacklist_base.txt")
args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

words_black_or = open(args.blacklist_path, "r").readlines()
words_black_or = [word.strip().lower() for word in words_black_or]
words_blacklist = ["[removed]", "[deleted]"]
words_blacklist.extend(words_black_or)
words_blacklist.remove("ass")

data = open(args.data_path, "r")

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
    tkns = list(gensim.utils.tokenize(instance))
    
    for i in range(2, len(tkns)):
        if tkns[i - 2] == tkns[i] and tkns[i - 1] == tkns[i]:
            reps = True
            break
    if reps:
        if debug:
            print(f"REPETITION SKIP: {instance}\n\n", file=sys.stderr)
        return True, 2

    eng_stopword_present = False
    for token in tkns:
        if token in eng_stopwords_dict:
            eng_stopword_present = True
            break

    if not eng_stopword_present:
        if debug:
            print(f"ENG STOPWORD NOT PRESENT SKIP: {instance}\n\n", file=sys.stderr)
        return True, 4

    doc = nlp(instance)
    if doc._.language['language']!="en":
        return True, 5
    
    return (False, -1)


counter_input = 0
counter_output = 0

final_dialogs = []

translations = {0: "blacklist", 1: "markup", 2: "repetition", 3: "http", 4: "structure", 5: "foreign"}

rules_num = 6
out = {}
for i in range(rules_num):
    out[i] = 0

if write_to_file:
    output_file = open(os.path.join(args.output_folder, args.preprocessed_file_name), "w")

for dialog in tqdm(data):
    counter_input+=1
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
        counter_output+=1
        if write_to_file:
            output_file.write(instance)
    else:
        out[i] += 1

if write_to_file:
    output_file.close()
data.close()

print(f"NUMBER OF DIALOGS BEFORE PREPROCESSING: {counter_input}")
print(f"NUMBER OF DIALOGS AFTER PREPROCESSING: {counter_output}")
print(f"REMOVED DIALOGS: {counter_input-counter_output}")
print(f"REMOVAL PER CATEGORY:")
for k in out.keys():
    print(f"Removed: {translations[k]}={out[k]}")
