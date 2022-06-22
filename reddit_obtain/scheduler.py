import subprocess
import json
import time
from threading import Thread


def run_dialog_retrieval(reddit_id, reddit_secret, process_id):
    def run_retrieve():
        subprocess.run(
            [
                "/mnt/rcala/mbti/bin/python",
                "/home/rcala/chatbot/reddit_obtain/dialog_retrieval.py",
                "--reddit_id",
                reddit_id,
                "--reddit_secret",
                reddit_secret,
                "--process_id",
                process_id,
            ]
        )

    return run_retrieve


BOTS_CONFIG_PATH = "/mnt/rcala/dialog_files/first_iteration/bots_config.json"
ERROR_FILE_TEMPLATE = "/mnt/rcala/dialog_files/first_iteration/error_logs/process_{}_error_log.txt"

bots_file = open(BOTS_CONFIG_PATH, "r")
bots_config = json.load(bots_file)

bot_threads = []

for bot_id in bots_config.keys():
    time.sleep(2)
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

    bots_file = open(BOTS_CONFIG_PATH, "r")
    bots_config = json.load(bots_file)

    for bot_id in bots_config.keys():
        time.sleep(1)
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
