import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F


data_path = "./dataset/processed/train.csv"
ds = pd.read_csv(data_path, dtype={"id": str})
print("loaded data")

max_token_len = 512
X = ds["comment_text"]
y = ds["score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
train_ds = pd.DataFrame({"comment_text": X_train, "score": y_train})
val_ds = pd.DataFrame({"comment_text": X_test, "score": y_test})

class ToxicityDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len, sample = 700_000):
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

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("created tokenizer")
train_dataset = ToxicityDataset(train_ds, tokenizer, max_token_len)
val_dataset = ToxicityDataset(val_ds, tokenizer, max_token_len, sample=None)
print("created datasets")

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

class ToxicityClassifier(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config["model_name"])
        # hidden layer
        self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        # classification layer
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config["n_labels"])
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        # dropout layer
        self.dropout = nn.Dropout()

    def forward(self, input_ids, attention_mask, label=None):
        # roberta model
        output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        # nerual network classfication layer
        pooled_output = self.hidden(pooled_output)
        # activation function
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = 0

        if label is not None:
            loss = self.loss_func(logits.view(-1, self.config["n_labels"]), label.view(-1, self.config['n_labels']))
            return loss, logits

    def training_step(self, batch, batch_index):
        loss, logits = self(**batch)
        self.log("train loss", loss, prog_bar = True, logger = True)
        return { "loss": loss, "predictions": logits, "label": batch["label"]}

    def validation_step(self, batch, batch_index):
        loss, logits = self(**batch)
        self.log("validation loss", loss, prog_bar = True, logger = True)
        return { "val_loss": loss, "predictions": logits, "label": batch["label"]}

    def prediction_step(self, batch, batch_index):
        _, logits = self(**batch)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["w_decay"])
        # optimizer = AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["w_decay"])
        total_steps = self.config["train_size"] / self.config["batch_size"]
        warmup_steps = math.floor(total_steps + self.config["warmup"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]

data_module = ToxicityDataModule(train_ds, val_ds, max_token_len, model_name = "roberta-base", batch_size=128)
data_module.setup()
print("setup data module")

config = {
    "model_name": "distilroberta-base",
    "n_labels": 1,
    "batch_size": 128,
    "lr": 1.5e-6,
    "warmup": 0.2,
    "train_size": len(data_module.train_dataloader()) ,
    "w_decay": 0.001,
    "n_epochs": 1
}

data_module = ToxicityDataModule(train_ds, val_ds, max_token_len, model_name = config["model_name"], batch_size = config["batch_size"])
data_module.setup()
print("setup data module")

model = ToxicityClassifier(config)
print(model.device)
model.to("cuda")

# idx=0
# input_ids = train_dataset.__getitem__(idx)['input_ids']
# attention_mask = train_dataset.__getitem__(idx)['attention_mask']
# label = train_dataset.__getitem__(idx)['label']
# model.cpu()
# loss, output = model(input_ids.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0), label.unsqueeze(dim=0))
# print(label.shape, output.shape, output)

if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=config["n_epochs"], num_sanity_val_steps=2, logger = True, enable_progress_bar = True, num_nodes = 1)
    print("created model. starting to fit the model...")
    trainer.fit(model, data_module)