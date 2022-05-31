import os
import numpy as np
import torch


from .lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from .gpt2_training.train_utils import load_model, boolean_string
from .generator_utils import top_filtering


def seed_all(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Generator:
    def __init__(self, args) -> None:
        seed_all(args["seed"])
        args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args["n_gpu"] = torch.cuda.device_count()

        self.tokenizer = GPT2Tokenizer.from_pretrained(args["model_name_or_path"])
        self.config = GPT2Config.from_json_file(
            os.path.join(args["model_name_or_path"], "config.json")
        )
        self.model = load_model(
            GPT2LMHeadModel(self.config), args["init_checkpoint"], args, verbose=True
        )
        self.eos = [self.tokenizer.encoder["<|endoftext|>"]]

        self.temperature = (
            args["temperature"] if args["temperature"] is not None else 0.9
        )
        self.top_k = args["top_k"] if args["top_k"] is not None else -1
        self.top_p = args["top_p"] if args["top_p"] is not None else 0.9
        self.model.eval()
        self.args = args
        self.reset()

    def reset(self):
        self.prev_input = None
        self.past_turns = []
        self.turns = 0

    def generate(self, user_prompt):
        with torch.no_grad():
            prev_conv = []
            for conv in self.past_turns:
                prev_conv += conv
                prev_conv += self.eos

            user_prompt = self.tokenizer.encode(user_prompt)
            prev_conv += user_prompt
            prev_conv += self.eos

            past = None
            if len(prev_conv) > self.args["max_seq_len"]:
                n = len(prev_conv) // self.args["max_seq_len"]
                for i in range(n):
                    self.prev_input = (
                        torch.LongTensor(
                            prev_conv[
                                i
                                * self.args["max_seq_len"] : (i + 1)
                                * self.args["max_seq_len"]
                            ]
                        )
                        .unsqueeze(0)
                        .to(self.args["device"])
                    )
                    logits, past = self.model(input_ids=self.prev_input, past=past)[:3]
                prev_conv = prev_conv[(i + 1) * self.args["max_seq_len"] :]

            self.prev_input = (
                torch.LongTensor(prev_conv).unsqueeze(0).to(self.args["device"])
            )
            logits, past = self.model(input_ids=self.prev_input, past=past)[:3]

            sent = []
            for i in range(128):

                if self.args["sampling"]:
                    logits = logits[:, -1, :] / self.temperature
                    logits = top_filtering(logits, top_k=self.top_k, top_p=self.top_p)
                    probs = torch.softmax(logits, dim=-1)
                    self.prev_input = torch.multinomial(probs, num_samples=1)
                    prev_word = self.prev_input.item()
                else:
                    logits = logits[:, -1, :]
                    prev_word = torch.topk(logits, 1, dim=1).indices.tolist()[0][0]
                    self.prev_input = (
                        torch.LongTensor([prev_word])
                        .unsqueeze(0)
                        .to(self.args["device"])
                    )

                if prev_word == self.eos[0]:
                    break
                sent.append(prev_word)

                logits, past = self.model(input_ids=self.prev_input, past=past)[:3]

            bot = self.tokenizer.decode(sent)
            print("Bot:", bot)

            self.turns += 1

            if self.args["sliding_window"]:
                if self.turns - 1 >= self.args["sliding_window_size"]:
                    for i in range(2, len(self.past_turns), 2):
                        self.past_turns[i - 2] = self.past_turns[i]
                        self.past_turns[i - 1] = self.past_turns[i + 1]
                    self.past_turns[-2] = user_prompt
                    self.past_turns[-1] = sent
                else:
                    self.past_turns += [user_prompt]
                    self.past_turns += [sent]
            else:
                if (
                    self.args["max_history"] != -1
                    and self.turns - 1 >= self.args["max_history"]
                    and (self.turns - 1) % self.args["max_history"] == 0
                ):
                    self.past_turns = []
                self.past_turns += [user_prompt]
                self.past_turns += [sent]

        return bot
