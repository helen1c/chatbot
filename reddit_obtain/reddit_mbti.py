import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAITS = ["introverted", "intuitive", "thinking", "perceiving"]
OPPOSITE_TRAITS = ["extroverted", "sensing", "feeling", "judging"]

def dialog_file_decorator(path):

    with open(path,"r") as file:

        for dialog_line in file:

            dialog_line = dialog_line.strip()

            dialog_parts = dialog_line.split(" EOS ")

            dialog_parts_eos = dialog_parts[:-1] 

            dialog_parts_tab = dialog_parts[-1].split("\t")

            dialog_parts = dialog_parts_eos + dialog_parts_tab

            bot_dialog_text = ""

            for idx in range(1,len(dialog_parts),2):

                bot_dialog_text += dialog_parts[idx][4:] + " "

            bot_dialog_text = bot_dialog_text.strip()

            data = {
                "bot_text": bot_dialog_text,
                "dialog": dialog_line
            }

            yield data

class MBTIDialogDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path


    def __iter__(self):

        dialog_file_iter = dialog_file_decorator(self.path)

        return dialog_file_iter
    

def collate_bert_mbti(samples, tokenizer):

    bot_texts = [sample["bot_text"] for sample in samples]
    dialogs = [sample['dialog'] for sample in samples]

    inputs = tokenizer(
        text=bot_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    return dialogs, bot_texts, inputs

def prepare_mbti_dialogs(path, batch_size, tokenizer):

    dataset = MBTIDialogDataset(path)

    collate_fun = lambda samples: collate_bert_mbti(samples, tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size,
        collate_fn=collate_fun,
    )

    return dataloader