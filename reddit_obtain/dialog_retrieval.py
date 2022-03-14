from distutils.command.config import config
import praw
import requests
import json
import time
import logging
import os
import argparse
from praw.models import MoreComments
import random
from random import randrange

SUBREDDITS_PATH = (
    "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/reddit_subreddits.txt"
)
NSFW_SUBREDDITS_PATH = (
    "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/nsfw_reddit_subreddits.txt"
)
DAILY_SECONDS = 86400
FIRST_EPOCH_TIMESTAMP_2018 = 1514764800
LAST_EPOCH_TIMESTAMP_2022 = 1646870400
REDDIT_NAME = "narval13068"
REDDIT_PASSWORD = "redditsifra123"
SECONDS_IN_HOUR = 3600
SUBMISSION_REDDIT_API_TEMP = "http://api.pushshift.io/reddit/search/submission/?subreddit={}&sort=asc&sort_type=created_utc&after={}&before={}"

parser = argparse.ArgumentParser()

parser.add_argument("--reddit_id", type=str, required=True)
parser.add_argument("--reddit_secret", type=str, required=True)
parser.add_argument(
    "--user_agent", type=str, default="reddit_extracting_dialog_data_agent_1"
)
parser.add_argument("--process_id", type=int, required=True)
parser.add_argument("--window_days", type=int, default=1)

# args = parser.parse_args(
#    "--reddit_id v7ayYQn4xD9XM0CyRWarQg --reddit_secret VPw_faXvaAS-ZCKFDF9bTkrPjZyY2Q --process_id 1".split()
# )

try:

    args = parser.parse_args()

    random.seed(int(args.process_id))

    already_chosen_json_path = "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/already_processed/already_processed_process_{}.json".format(
        args.process_id
    )

    dialog_output_path = "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/dialogs/reddit_dialogs_process_{}.tsv".format(
        args.process_id
    )

    error_output_path = "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/error_logs/process_{}_error_log.txt".format(
        args.process_id
    )

    logs_output_path = "/Users/arvencala/Desktop/Faculty/Rektor/Datasets/Reddit/logs/process_{}_log.txt".format(
        args.process_id
    )

    logging.basicConfig(
        filename=logs_output_path,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    epoch_step = DAILY_SECONDS * args.window_days

    checkpoint_time = time.time()
    if os.path.exists(already_chosen_json_path) is False:
        config_file = open(already_chosen_json_path, "w")
        config_file.write("{}")
        config_file.close()
    already_processed = json.load(open(already_chosen_json_path, "r"))
    first = True
    subreddits = open(SUBREDDITS_PATH, "r").read().split("\n")
    nsfw_subreddits = open(NSFW_SUBREDDITS_PATH, "r").read().split("\n")
    filtered_subreddits = []
    for subreddit in subreddits:
        if subreddit in nsfw_subreddits:
            continue
        filtered_subreddits += [subreddit]
    subreddit_num = len(filtered_subreddits)

    logger.info("START")

    while True:

        if time.time() - checkpoint_time >= SECONDS_IN_HOUR or first:

            checkpoint_time = time.time()

            reddit = praw.Reddit(
                client_id=args.reddit_id,
                client_secret=args.reddit_secret,
                user_agent=args.user_agent,
            )
            auth = requests.auth.HTTPBasicAuth(args.reddit_id, args.reddit_secret)

            data = {
                "grant_type": "password",
                "username": REDDIT_NAME,
                "password": REDDIT_PASSWORD,
            }
            headers = {"User-Agent": "Reddit dialog model"}
            res = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers=headers,
            )

            logger.info("Obtained Reddit API token")

            first = False

        subreddit_idx = randrange(subreddit_num)
        subreddit_name = filtered_subreddits[subreddit_idx]

        created_utc_after = randrange(
            FIRST_EPOCH_TIMESTAMP_2018, LAST_EPOCH_TIMESTAMP_2022, epoch_step
        )
        created_utc_before = created_utc_after + epoch_step

        already_processed_key = subreddit_name + "-" + str(created_utc_after)

        if already_processed_key in already_processed:
            continue

        time.sleep(2)

        logger.info(already_processed_key)

        request_url = SUBMISSION_REDDIT_API_TEMP.format(
            subreddit_name, created_utc_after, created_utc_before
        )

        logger.info(request_url)

        res = requests.get(request_url, headers=headers)

        logger.info(res)

        if res.status_code != 200:
            continue

        logger.info(
            "Total number of submissions: {}".format(str(len(res.json()["data"])))
        )
        total_top_comments = 0
        total_dialogs = 0

        all_dialogs = []

        for submission in res.json()["data"]:

            submission_id = submission["id"]
            praw_submission = reddit.submission(submission_id)

            try:
                if praw_submission.over_18:
                    continue
            except Exception as e:
                continue

            for top_level_comment in praw_submission.comments:

                total_top_comments += 1

                if isinstance(top_level_comment, MoreComments):
                    continue

                bottom = False
                new_dialog = [" ".join(top_level_comment.body.split())]
                inner_reply = top_level_comment

                while bottom == False:

                    if len(inner_reply.replies) == 0:
                        bottom = True
                        continue

                    next_inner_reply = None
                    for reply in inner_reply.replies:
                        if not isinstance(reply, MoreComments):
                            next_inner_reply = reply
                            break

                    if next_inner_reply is not None:
                        new_dialog += [" ".join(next_inner_reply.body.split())]
                        inner_reply = next_inner_reply
                    else:
                        bottom = True

                if len(new_dialog) == 1:
                    continue

                all_dialogs += [new_dialog]

        total_dialogs += len(all_dialogs)
        dialog_output_file = open(dialog_output_path, "a")

        for dialog in all_dialogs:
            zero = True
            dialog_line = ""
            for idx, dialog_utterance in enumerate(dialog[:-1]):

                if zero:
                    dialog_line += "0.0 "
                    dialog_line += dialog_utterance + " "
                    zero = False
                else:
                    dialog_line += "1.0 "
                    dialog_line += dialog_utterance + " "
                    zero = True

                if idx != len(dialog[:-1]) - 1:
                    dialog_line += "EOS "
                else:
                    dialog_line = dialog_line[:-1]

            dialog_line += "\t"

            if zero:
                dialog_line += "0.0 " + dialog[-1]
            else:
                dialog_line += "1.0 " + dialog[-1]

            dialog_output_file.write(already_processed_key + "\t" + dialog_line + "\n")

        dialog_output_file.flush()
        dialog_output_file.close()

        logger.info("Total number of top comments: {}".format(total_top_comments))
        logger.info("Total number of dialogs: {}".format(total_dialogs))

        already_processed[already_processed_key] = 1
        json.dump(already_processed, open(already_chosen_json_path, "w"))

except Exception as e:
    open(error_output_path, "w").write("Error\n")
    logger.info(str(e))
    logger.info("END")

