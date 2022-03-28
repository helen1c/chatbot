import json

BOTS_CONFIG_PATH = "/storage/rcala/Reddit/bots_config.json"
SUBREDDITS_PATH = "/storage/rcala/Reddit/reddit_subreddits.txt"
NSFW_SUBREDDITS_PATH = "/storage/rcala/Reddit/nsfw_reddit_subreddits.txt"
DAILY_SECONDS = 86400
FIRST_EPOCH_TIMESTAMP_2018 = 1514764800
LAST_EPOCH_TIMESTAMP_2022 = 1646870400
WINDOW_DAYS = 1
NUM_PERIODS = int(
    (LAST_EPOCH_TIMESTAMP_2022 - FIRST_EPOCH_TIMESTAMP_2018)
    / (WINDOW_DAYS * DAILY_SECONDS)
)
NUM_BOTS = 350
ALREADY_PROCESSED_TEMPLATE = (
    "/storage/rcala/Reddit/already_processed/already_processed_process_{}.json"
)

bots_file = open(BOTS_CONFIG_PATH, "r")
bots_config = json.load(bots_file)

orig_subreddits = open(SUBREDDITS_PATH, "r").read().split("\n")
orig_num_subreddits = len(orig_subreddits)
nsfw_subreddits = open(NSFW_SUBREDDITS_PATH, "r").read().split("\n")

for bot_id in bots_config:

    subreddits = orig_subreddits[
        int((int(bot_id) - 1) * orig_num_subreddits / NUM_BOTS) : int(
            int(bot_id) * orig_num_subreddits / NUM_BOTS
        )
    ]

    filtered_subreddits = []
    for subreddit in subreddits:
        if subreddit in nsfw_subreddits:
            continue
        filtered_subreddits += [subreddit]
    subreddit_num = len(filtered_subreddits)

    max_num = subreddit_num * NUM_PERIODS

    already_processed_file = open(ALREADY_PROCESSED_TEMPLATE.format(bot_id), "r")
    already_processed = json.load(already_processed_file)

    print(
        "Bot {}: {}/{}, {:.2f}%".format(
            bot_id,
            len(already_processed),
            max_num,
            100 * len(already_processed) / max_num,
        )
    )
