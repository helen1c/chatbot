import json
import time

COMBINED_DIALOG_FILE_PATH = "/home/rcala/chatbot/reddit_obtain/Reddit/reddit_dialogs.tsv"
DIALOG_FILE_TEMPLATE_PATH = "/home/rcala/chatbot/reddit_obtain/Reddit/dialogs/reddit_dialogs_process_{}.tsv"
BOTS_CONFIG_PATH = "/home/rcala/chatbot/reddit_obtain/Reddit/bots_config.json"

#safety, comment if sure 
time.sleep(10)

dialog_file = open(COMBINED_DIALOG_FILE_PATH,"w")

bots_file = open(BOTS_CONFIG_PATH, "r")
bots_config = json.load(bots_file)

for bot_id in bots_config:
    lines = open(DIALOG_FILE_TEMPLATE_PATH.format(bot_id),"r").read().splitlines()
    for line in lines:
        items = line.split('\t')
        dialog_file.write("\t".join(items[1:])+"\n")
    dialog_file.flush()

dialog_file.close()

    