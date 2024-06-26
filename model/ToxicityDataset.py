import pandas as pd
import torch
from torch.utils.data import Dataset

class ToxicityDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len, sample = 10_000):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = int(max_token_len)
        self.sample = sample
        self.__prepare_data()

    def __prepare_data(self):
        if self.sample is not None:
            toxic = self.data.loc[self.data["score"] > 0]
            not_toxic = self.data.loc[self.data["score"] == 0]
            self.data = pd.concat([toxic, not_toxic.sample(self.sample, random_state=53)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index > len(self.data):
            raise IndexError("Index out of bound: ", f"index: {index}, len: {len(self.data)}")
        item = self.data.iloc[index]
        comment = str(item.comment_text)
        score = torch.tensor(item.iloc[1], dtype=torch.float32)
        # score = torch.FloatTensor(item[["score"]])

        tokens = self.tokenizer.encode_plus(comment,
                                           add_special_tokens=True,
                                           return_tensors="pt",
                                           truncation=True,
                                           max_length=self.max_token_len,
                                           padding="max_length",
                                           return_attention_mask=True)

        return {"input_ids": tokens.input_ids.flatten(), "attention_mask": tokens.attention_mask.flatten(), "label": score}