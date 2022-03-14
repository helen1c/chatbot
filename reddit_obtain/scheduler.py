from re import T
import subprocess
import json
import time
from threading import Thread

MAX_NODES = 10


def run_dialog_retrieval(reddit_id, reddit_secret, process_id):
    def run_retrieve():
        subprocess.run(
            [
                "/Users/arvencala/miniconda3/bin/python",
                "dialog_retrieval.py",
                "--reddit_id",
                reddit_id,
                "--reddit_secret",
                reddit_secret,
                "--process_id",
                process_id,
            ]
        )

    return run_retrieve


BOTS_CONFIG_PATH = (
    "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/bots_config.json"
)
ERROR_FILE_TEMPLATE = "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/error_logs/process_{}_error_log.txt"

bots_file = open(BOTS_CONFIG_PATH, "r")
bots_config = json.load(bots_file)

bot_threads = []

for bot_id in bots_config.keys():
    if int(bot_id) > MAX_NODES:
        continue
    time.sleep(5)
    empty_file = open(ERROR_FILE_TEMPLATE.format(bot_id), "w")
    empty_file.close()
    bot_threads += [
        Thread(
            target=run_dialog_retrieval(
                reddit_id=bots_config[bot_id][0],
                reddit_secret=bots_config[bot_id][1],
                process_id=bot_id,
            )
        )
    ]
    bot_threads[int(bot_id) - 1].start()

while True:

    for bot_id in bots_config.keys():
        if int(bot_id) > MAX_NODES:
            continue
        time.sleep(2)
        error_file = open(ERROR_FILE_TEMPLATE.format(bot_id), "r")
        lines = error_file.readlines()
        error_file.close()
        if len(lines) > 0 and lines[0].strip() == "Error":
            empty_file = open(ERROR_FILE_TEMPLATE.format(bot_id), "w")
            empty_file.close()
            bot_threads[int(bot_id) - 1] = Thread(
                target=run_dialog_retrieval(
                    reddit_id=bots_config[bot_id][0],
                    reddit_secret=bots_config[bot_id][1],
                    process_id=bot_id,
                )
            )
            bot_threads[int(bot_id) - 1].start()

