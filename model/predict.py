import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
import numpy as np
from ToxicityDataModule import ToxicityDataModule, max_token_len
from ToxicityClassifier import ToxicityClassifier
import json

def main():
    data_path = "./dataset/processed/train.csv"
    ds = pd.read_csv(data_path, dtype={"id": str})
    print("loaded data")

    X = ds["comment_text"]
    y = ds["score"]
    X, _, y, _ = train_test_split(X, y, test_size=0.7, random_state=103)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
    train_ds = pd.DataFrame({"comment_text": X_train, "score": y_train})
    val_ds = pd.DataFrame({"comment_text": X_test, "score": y_test})


    tmp_data_module = ToxicityDataModule(train_ds, val_ds, max_token_len, model_name = "roberta-base", batch_size=128)

    tmp_data_module.setup()
    print("setup data module")

    config = json.load(open("config.json"))

    data_module = ToxicityDataModule(train_ds, val_ds, max_token_len, model_name = config["model_name"], batch_size = config["batch_size"])
    data_module.setup()

    trainer = pl.Trainer(max_epochs=config["n_epochs"], num_sanity_val_steps=2, logger = True, enable_progress_bar = True, num_nodes = 1, precision=16) # change to 16-mixed in the future
    model = ToxicityClassifier(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torch.set_float32_matmul_precision("medium")

    model.load_state_dict(torch.load("./trained/v1.pth"))

    def predict_toxicity(model, data_module):
        predictions = trainer.predict(model, datamodule=data_module)
        flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
        return flattened_predictions

    predictions = predict_toxicity(model, data_module)
    print(predictions)

if __name__ == "__main__":
    main()

