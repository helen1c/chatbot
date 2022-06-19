import os
import numpy as np
import torch
import argparse
import math

from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import load_model, boolean_string

os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=1)
parser.add_argument('--model_name_or_path', type=str,
                    help='trained model name or path to local chesckpoint')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--fp16", type=boolean_string, default=False)

args = parser.parse_args("--model_name_or_path /mnt/rcala/dialogpt/models/medium --init_checkpoint /mnt/rcala/dialogpt/models/medium/medium_ft.pkl".split())

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed) 

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
config = GPT2Config.from_json_file(
    os.path.join(args.model_name_or_path, 'config.json'))
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,
                  args, verbose=True)
model.eval()

with open("/home/rcala/chatbot/DialoGPT/mbti_test/mbti_test.txt","r") as f:
    all_mbti_prompts_without_labels = f.read().split("\n")
mbti_prompts_with_labels = []

for mbti_prompt_without_label in all_mbti_prompts_without_labels:
    with torch.no_grad():

        mbti_prompt_without_label = [mbti_prompt_without_label]

        inputs = tokenizer(
            text=mbti_prompt_without_label,
            return_tensors="pt",
        )

        inputs = {k: v.type(torch.long).to(args.device) for k, v in inputs.items()}

        trait_indices = tokenizer.encode(
            " agree", add_special_tokens=False
        )
        opposite_trait_indices = tokenizer.encode(
            " disagree", add_special_tokens=False
        )

        if len(trait_indices) > len(opposite_trait_indices):
            diff_len = len(trait_indices) - len(opposite_trait_indices)
            for _ in range(diff_len):
                opposite_trait_indices += [tokenizer.eos_token_id]

        if len(opposite_trait_indices) > len(trait_indices):
            diff_len = len(opposite_trait_indices) - len(trait_indices)
            for _ in range(diff_len):
                trait_indices += [tokenizer.eos_token_id]

        new_generate_len = len(trait_indices)

        added_token_id = 50256

        for idx in range(new_generate_len):

            tokens_logits, past = model(input_ids = inputs['input_ids'])[:3]
            token_logits = tokens_logits[0, [-1], :]

            decoding_indices = []
            if idx == 0:
                decoding_indices = [trait_indices[idx]] + [opposite_trait_indices[idx]]
            if idx > 0:
                if added_token_id == trait_indices[idx - 1]:
                    decoding_indices += [trait_indices[idx]]
                if added_token_id == opposite_trait_indices[idx - 1]:
                    decoding_indices += [opposite_trait_indices[idx]]

            decoding_indices = torch.tensor(decoding_indices)
            decoding_indices_oh = torch.nn.functional.one_hot(
                decoding_indices, tokenizer.vocab_size
            )

            mask_unimportant = torch.sum(decoding_indices_oh, 0) == 0
            mask_unimportant = mask_unimportant.repeat(token_logits.shape[0], 1)
            token_logits[mask_unimportant] = -math.inf
            added_token_id = torch.topk(token_logits, 1, dim=1).indices.tolist()[0][0]
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor([[added_token_id]]).to(args.device)), dim=1
            )

        predicted_prompt = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )
        mbti_prompts_with_labels += [predicted_prompt]

for filled_prompt in mbti_prompts_with_labels:
    print(filled_prompt)



