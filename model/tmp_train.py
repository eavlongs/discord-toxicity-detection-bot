import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import os

# Load the dataset
data_path = "./dataset/processed/train.csv"
df = pd.read_csv(data_path, dtype={"id": str})

# Preprocess the dataset
df['comment_text'] = df['comment_text'].astype(str)
df['score'] = df['score'].astype(int)

not_toxic = df.loc[df["score"] == 0]
level_2_toxic = df.loc[df["score"] == 2]
level_1_toxic = df.loc[df["score"] == 1]
higher_than_2_toxic = df.loc[df["score"] > 2]
nine_and_ten = df.loc[df["score"] >= 9]

# resample the data, because of imbalance. See test_sample for visualization
df = pd.concat([not_toxic.sample(70_000, random_state=53), level_2_toxic.sample(65_000, random_state=53), level_1_toxic, higher_than_2_toxic, nine_and_ten])

# Initialize the 'target' column with NaN values
df['target'] = np.nan

# Assign 0 to 'target' for non-toxic scores
df.loc[df['score'] < 4, 'target'] = 0

# Assign 1 to 'target' for toxic scores
df.loc[df['score'] >= 4, 'target'] = 1

# Split the DataFrame into not_toxic_at_all and toxic
not_toxic_at_all = df.loc[df['target'] == 0]
toxic = df.loc[df['target'] == 1]

df = pd.concat([not_toxic_at_all, toxic])

# shuffle data
df.sample(frac=1).reset_index(drop=True)

# df, _ = train_test_split(df, test_size=0.99999, random_state=8)

# df["score"] = df["score"].apply(lambda x: 3 if x == 4 else x)

# print(np.unique(df['score']))

# exit()

# Split the dataset into training and validation sets
df, _ = train_test_split(df, test_size=0.75, random_state=42)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# print(train_df.shape)

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['comment_text']
        # score = self.df.iloc[index]['score']
        target = self.df.iloc[index]['target']

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
            'targets': torch.tensor([target], dtype=torch.long)
        }

# Parameters
MAX_LEN = 192
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
<<<<<<< HEAD
EPOCHS = 1
LEARNING_RATE = 3e-5
=======
EPOCHS = 4
LEARNING_RATE = 1e-5
>>>>>>> f159269b5a845d797e3112b3a0b4451da5f2db87

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

# Create datasets
training_set = ToxicityDataset(train_df, tokenizer, MAX_LEN)
validation_set = ToxicityDataset(val_df, tokenizer, MAX_LEN)

# Create data loaders
train_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, shuffle=False)

class ToxicityClassifer(torch.nn.Module):
    def __init__(self):
        super(ToxicityClassifer, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
<<<<<<< HEAD
        pooler = torch.sigmoid(pooler)
=======
        pooler = torch.nn.ReLU()(pooler)
>>>>>>> f159269b5a845d797e3112b3a0b4451da5f2db87
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# Model definition
model = ToxicityClassifer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# to solve data imbalance problem
# class_weights = compute_class_weight('balanced', classes=np.unique(df['score']), y=df['score'])
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# print(class_weights)

# we are using sparse categorial crossentropy, because its for multi-class classification
# loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params = model.parameters(), lr=LEARNING_RATE)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Checkpoint directory
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

best_accuracy = 0

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

# Training function
def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    start_time = time.time()
    print(f"Training Epoch: {epoch}")
    print(f"Training Dataset Size: {len(train_loader)}")
    for _, data in tqdm(enumerate(train_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val = torch.sigmoid(outputs)
        big_idx = (big_val > 0.5).float()
        n_correct += (big_idx == targets).sum().item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training Loss per 100 steps: {loss_step}")
            print(f"Time elapsed: {time.time() - start_time:.2f}s")
            start_time = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {n_correct / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = n_correct / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")

    return

# Validation function
def valid(model, testing_loader):
    model.eval()
    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    start_time = time.time()
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val = torch.sigmoid(outputs)
            big_idx = (big_val > 0.5).float()
            n_correct += (big_idx == targets).sum().item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Time elapsed: {time.time() - start_time:.2f}s")
                start_time = time.time()
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = n_correct / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu

if __name__ == "__main__":
    # Uncomment when training
    for epoch in range(EPOCHS):
        train(epoch)
        acc = valid(model, val_loader)
        is_best = acc > best_accuracy
        best_accuracy = max(acc, best_accuracy)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_accuracy': best_accuracy,
        }
        save_checkpoint(checkpoint, is_best, filename=os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    output_model_file = './trained/v9.pth'

    model_to_save = model
    torch.save(model_to_save.state_dict(), output_model_file)

    # Uncomment when validating
    acc = valid(model, val_loader)
    print("Accuracy on test data = %0.2f%%" % acc)

    model_to_save = model
    torch.save(model_to_save.state_dict(), output_model_file)