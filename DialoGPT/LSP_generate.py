import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse

from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import load_model, boolean_string

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k, top-p (nucleus) aHnd/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -float("Inf"),
            logits,
        )
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits = sorted_logits.masked_fill_(
            sorted_indices_to_remove, filter_value
        )
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    return logits


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_history", type=int, default=5)
parser.add_argument("--sliding_window", type=bool, default=True)
parser.add_argument("--sliding_window_size", type=int, default=5)
parser.add_argument("--sampling", type=bool, default=True)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="trained model name or path to local chesckpoint",
)
parser.add_argument("--max_seq_len", type=int, default=128)
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--fp16", type=boolean_string, default=False)

args = parser.parse_args(
    "--model_name_or_path DialoGPT\models\medium\medium --init_checkpoint DialoGPT\models\GPT2.1e-05.4.1gpu.2022-04-30115722_introverted_550k_structure\GP2-finetune-step-186129.pkl".split()
)

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, "config.json"))
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint, args, verbose=True)



eos = [tokenizer.encoder["<|endoftext|>"]]

temperature = 0.9
top_k = -1
top_p = 0.9



model.eval()

prev_input = None
past_turns = []
turns = 0

while True:
    with torch.no_grad():
        # input and update B's utterance
        user = input("User: ")

        if user == "quit":
            "stop talking!"
            break

        if user == "reset":
            past_turns = []
            prev_input = None
            turns = 0
            continue

        prev_conv = []
        for conv in past_turns:
            prev_conv += conv
            prev_conv += eos

        user = tokenizer.encode(user)
        prev_conv += user
        prev_conv += eos

        past = None
        if len(prev_conv) > args.max_seq_len:
            n = len(prev_conv) // args.max_seq_len
            for i in range(n):
                prev_input = (
                    torch.LongTensor(
                        prev_conv[i * args.max_seq_len : (i + 1) * args.max_seq_len]
                    )
                    .unsqueeze(0)
                    .to(args.device)
                )
                logits, past = model(input_ids=prev_input, past=past)[:3]
            prev_conv = prev_conv[(i + 1) * args.max_seq_len :]

        prev_input = torch.LongTensor(prev_conv).unsqueeze(0).to(args.device)
        logits, past = model(input_ids=prev_input, past=past)[:3]

        sent = []
        for i in range(128):

            if args.sampling:
                logits = logits[:, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                prev_input = torch.multinomial(probs, num_samples=1)
                prev_word = prev_input.item()
            else:
                logits = logits[:, -1, :]
                prev_word = torch.topk(logits, 1, dim=1).indices.tolist()[0][0]
                prev_input = torch.LongTensor([prev_word]).unsqueeze(0).to(args.device)

            if prev_word == eos[0]:
                break
            sent.append(prev_word)

            logits, past = model(input_ids=prev_input, past=past)[:3]

        bot = tokenizer.decode(sent)
        print("Bot:", bot)

        turns += 1

        if args.sliding_window:
            if turns - 1 >= args.sliding_window_size:
                for i in range(2, len(past_turns), 2):
                    past_turns[i - 2] = past_turns[i]
                    past_turns[i - 1] = past_turns[i + 1]
                past_turns[-2] = user
                past_turns[-1] = sent
            else:
                past_turns += [user]
                past_turns += [sent]
        else:
            if (
                args.max_history != -1
                and turns - 1 >= args.max_history
                and (turns - 1) % args.max_history == 0
            ):
                past_turns = []
            past_turns += [user]
            past_turns += [sent]
