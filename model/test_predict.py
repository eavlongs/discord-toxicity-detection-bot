import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model_path = './trained-temp/v3.pth'  # Adjust the path if necessary
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=11)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Dataset preparation
class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=192):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index % 100 == 0:  # Adjust the interval as needed
            print(f"Processing item {index}/{len(self.df)}")
        text = self.df.iloc[index]['comment_text']
        score = self.df.iloc[index]['score']
        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', return_token_type_ids=True, truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(score, dtype=torch.long)
        }

# Load your dataset
data_path = "./dataset/processed/train.csv"  # Adjust the path if necessary
df = pd.read_csv(data_path, dtype={"id": str})
df['comment_text'] = df['comment_text'].astype(str)
df['score'] = df['score'].astype(int)
_, val_df = train_test_split(df, test_size=0.1, random_state=42)  # Adjust split ratio if necessary

# Create validation dataset and loader
validation_set = ToxicityDataset(val_df, tokenizer)
val_loader = DataLoader(validation_set, batch_size=8, shuffle=False)  # Adjust batch size if necessary

# Prediction function
def get_raw_predictions(loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            outputs = model(input_ids=ids, attention_mask=mask)
            predictions.extend(outputs.logits.cpu().detach().numpy().tolist())
    return predictions

def predict_single_comment(comment, tokenizer, model, device, max_len=192):
    model.eval()  # Ensure the model is in evaluation mode
    inputs = tokenizer.encode_plus(
        comment, None, add_special_tokens=True, max_length=max_len,
        padding='max_length', return_token_type_ids=True, truncation=True
    )
    
    ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
    mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
    
    return probabilities.cpu().numpy()

# Example usage
comment = "fuck you you motherfucker"
probabilities = predict_single_comment(comment, tokenizer, model, device)
print(probabilities)

# # Get raw predictions
# raw_predictions = get_raw_predictions(val_loader)
# print(raw_predictions)