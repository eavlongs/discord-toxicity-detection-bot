import json
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
from ToxicityDataModule import ToxicityDataModule, max_token_len
from ToxicityClassifier import ToxicityClassifier
from ToxicityDataset import ToxicityDataset

data_path = "./dataset/processed/train.csv"
ds = pd.read_csv(data_path, dtype={"id": str})
print("loaded data")

X = ds["comment_text"]
y = ds["score"]
X, _, y, _ = train_test_split(X, y, test_size=0.7, random_state=103)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
train_ds = pd.DataFrame({"comment_text": X_train, "score": y_train})
val_ds = pd.DataFrame({"comment_text": X_test, "score": y_test})


model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("created tokenizer")
train_dataset = ToxicityDataset(train_ds, tokenizer, max_token_len)
val_dataset = ToxicityDataset(val_ds, tokenizer, max_token_len, sample=None)
print("created datasets")

tmp_data_module = ToxicityDataModule(train_ds, val_ds, max_token_len, model_name = "roberta-base", batch_size=128)
tmp_data_module.setup()
print("setup data module")

config = {
    "model_name": "distilroberta-base",
    "n_labels": 1,
    "batch_size": 32,
    "lr": 1e-4,
    "warmup": 0.1,
    "train_size": len(tmp_data_module.train_dataloader()),
    "w_decay": 0.001,
    "n_epochs": 3
}

json.dump(config, open("config.json", "w"))

data_module = ToxicityDataModule(train_ds, val_ds, max_token_len, model_name = config["model_name"], batch_size = config["batch_size"])
data_module.setup()
print("setup data module")

model = ToxicityClassifier(config)
model.to("cuda")

# if __name__ == "__main__":
#     trainer = pl.Trainer(max_epochs=config["n_epochs"], num_sanity_val_steps=2, logger = True, enable_progress_bar = True, num_nodes = 1)
#     print("created model. starting to fit the model...")
#     trainer.fit(model, data_module)