import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ToxicityDataset import ToxicityDataset

max_token_len = 512

class ToxicityDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, max_token_len, model_name, batch_size):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def setup(self, stage = "fit"):
        if stage == "fit":
            self.train_ds = ToxicityDataset(self.train_data, self.tokenizer, max_token_len)
            self.val_ds = ToxicityDataset(self.val_data, self.tokenizer, max_token_len, sample=None)
        if stage == "predict":
            self.val_ds = ToxicityDataset(self.val_data, self.tokenizer, max_token_len, sample=None)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size, num_workers=4, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size = self.batch_size, num_workers=4, shuffle=False,  persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, batch_size = self.batch_size, num_workers=4, shuffle=False, persistent_workers=True)