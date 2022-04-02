import torch
from torch.utils.data import DataLoader

TRAITS = ["introverted", "intuitive", "thinking", "perceiving"]
OPPOSITE_TRAITS = ["extroverted", "sensing", "feeling", "judging"]

def prepare_bot_dialogs(path):

    with open(path,'r') as dialogs_file:
        dialog_lines = dialogs_file.read().splitlines()

    bot_dialog_texts = []

    for dialog_line in dialog_lines:

        dialog_parts = dialog_line.split("\t")

        bot_dialog_text = ""

        for idx in range(1,len(dialog_parts),2):

            bot_dialog_text += dialog_parts[idx][4:] + " "

        bot_dialog_text = bot_dialog_text.strip()
        bot_dialog_texts += [bot_dialog_text]
    
    return dialog_lines, bot_dialog_texts


class MBTIDialogDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        super().__init__()
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        item = {"text": self.texts[idx]}

        return item

def collate_bert_mbti(samples, tokenizer):

    texts = [sample["text"] for sample in samples]

    inputs = tokenizer(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    return texts, inputs

def prepare_mbti_dialogs(path, batch_size, tokenizer):

    dialog_lines, bot_dialog_texts = prepare_bot_dialogs(path)

    collate_fun = lambda samples: collate_bert_mbti(samples, tokenizer=tokenizer)

    dataset = MBTIDialogDataset(bot_dialog_texts)

    dataloader = DataLoader(
        dataset,
        batch_size,
        collate_fn=collate_fun,
    )

    return dialog_lines, dataloader