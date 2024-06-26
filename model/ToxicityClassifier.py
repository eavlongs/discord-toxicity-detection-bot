import torch
import math
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

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

    def predict_step(self, batch, batch_index):
        _, logits = self(**batch)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["w_decay"])
        # optimizer = AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["w_decay"])
        total_steps = self.config["train_size"] / self.config["batch_size"]
        warmup_steps = math.floor(total_steps + self.config["warmup"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]