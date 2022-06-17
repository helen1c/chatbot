import torch
import torch.nn.functional as F
from .lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
from .gpt2_training.train_utils import load_model

available_generators = {
    0: {
        "model_id": 0,
        "model_name": "Extrovert",
        "model_name_or_path": "DialoGPT\models\medium\medium",
        "init_checkpoint": "DialoGPT\models\checkpoints\extrovert\extroverted_GP2-finetune-step-206382.pkl",
    },
    1: {
        "model_id": 1,
        "model_name": "Introvert",
        "model_name_or_path": "DialoGPT\models\medium\medium",
        "init_checkpoint": "DialoGPT\models\checkpoints\introvert\introverted_GP2-finetune-step-253076.pkl",
    },
    2: {
        "model_id": 2,
        "model_name": "Intuitive",
        "model_name_or_path": "DialoGPT\models\medium\medium",
        "init_checkpoint": "DialoGPT\models\checkpoints\intuitive\intuitive_GP2-finetune-step-322030.pkl",
    },
    3: {
        "model_id": 3,
        "model_name": "Feeling",
        "model_name_or_path": "DialoGPT\models\medium\medium",
        "init_checkpoint": "DialoGPT\models\checkpoints\feeling\feeling_GP2-finetune-step-194997.pkl",
    },
}


def load_generator_models(args, num_of_mod=None):

    if num_of_mod is None:
        num_of_mod = len(available_generators.keys())

    models = {}

    loaded = 0
    keys = available_generators.keys()
    for key in keys:
        current_dict = {}
        current_entry = available_generators[key]
        current_dict["model_id"] = current_entry["model_id"]
        current_dict["model_name"] = current_entry["model_name"]
        tokenizer = GPT2Tokenizer.from_pretrained(current_entry["model_name_or_path"])
        config = GPT2Config.from_json_file(
            os.path.join(current_entry["model_name_or_path"], "config.json")
        )

        print(f"Loading model: {str(current_dict)}")

        model = load_model(
            GPT2LMHeadModel(config),
            current_entry["init_checkpoint"],
            args,
            verbose=True,
        )

        current_dict["model"] = model
        current_dict["tokenizer"] = tokenizer
        current_dict["config"] = config

        models[key] = current_dict
        loaded += 1
        if loaded >= num_of_mod:
            break

    return models


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
