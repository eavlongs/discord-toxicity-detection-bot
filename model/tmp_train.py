import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import time

# Load the dataset
data_path = "./dataset/processed/train.csv"
df = pd.read_csv(data_path, dtype={"id": str})

# Preprocess the dataset
df['comment_text'] = df['comment_text'].astype(str)
df['score'] = df['score'].astype(int)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.9, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['comment_text']
        score = self.df.iloc[index]['score']

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(score, dtype=torch.long)
        }

# Parameters
MAX_LEN = 192
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-4

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create datasets
training_set = ToxicityDataset(train_df, tokenizer, MAX_LEN)
validation_set = ToxicityDataset(val_df, tokenizer, MAX_LEN)

# Create data loaders
train_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, shuffle=False)

# Model definition
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=11)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training function
def train(epoch):
    model.train()
    start_time = time.time()
    for step, data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f'Epoch: {epoch}, Step: {step}/{len(train_loader)}, Loss: {loss.item()}, Time elapsed: {elapsed_time:.2f}s')
            start_time = time.time()

# Validation function
def validate():
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.argmax(outputs.logits, dim=1).cpu().detach().numpy().tolist())

    accuracy = accuracy_score(fin_targets, fin_outputs)
    f1 = f1_score(fin_targets, fin_outputs, average='weighted')
    precision = precision_score(fin_targets, fin_outputs, average='weighted')
    recall = recall_score(fin_targets, fin_outputs, average='weighted')

    return accuracy, f1, precision, recall

# Training loop
for epoch in range(EPOCHS):
    train(epoch)
    accuracy, f1, precision, recall = validate()
    print(f'Epoch: {epoch}, Validation Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')

torch.save(model.state_dict(), './trained/v2.pth')