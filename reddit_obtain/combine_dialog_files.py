import json
import time
from tqdm import tqdm

COMBINED_DIALOG_FILE_PATH = "/mnt/rcala/dialog_files/first_iteration/reddit_dialogs.tsv"
DIALOG_FILE_TEMPLATE_PATH = (
    "/mnt/rcala/dialog_files/first_iteration/dialogs/reddit_dialogs_process_{}.tsv"
)
BOTS_CONFIG_PATH = "/mnt/rcala/dialog_files/first_iteration/bots_config.json"

# safety, comment if sure
# time.sleep(10)

dialog_file = open(COMBINED_DIALOG_FILE_PATH, "w")

bots_file = open(BOTS_CONFIG_PATH, "r")
bots_config = json.load(bots_file)

for bot_id in tqdm(bots_config):
    lines = open(DIALOG_FILE_TEMPLATE_PATH.format(bot_id), "r", errors='ignore')
    for line in lines:
        line = line[:-1]
        items = line.split("\t")
        dialog_file.write("\t".join(items[1:]) + "\n")
    dialog_file.flush()

dialog_file.close()
