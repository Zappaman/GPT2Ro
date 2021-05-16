"""
Pytorch Dataset class for fine-tuning a GPT2 model
for the Romanian language
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.random_access_file_reader import RandomAccessFileReader

from transformers import GPT2Tokenizer


class RoDataset(Dataset):
    def __init__(self, tokenizer_path_prefix: str, src_file: str, max_length: int = 1024):
        merge_txt = f"{tokenizer_path_prefix}merges.txt"
        vocab_json = f"{tokenizer_path_prefix}vocab.json"
        tokenizer = GPT2Tokenizer(
            vocab_json,
            merge_txt,
            pad_token="<pad>",
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rf = RandomAccessFileReader(src_file)

    def __len__(self):
        return self.rf.filelen

    def __getitem__(self, i):
        line = self.rf.getLine(i)
        return self.tokenizer(line, max_length=self.max_length, truncation=True)
