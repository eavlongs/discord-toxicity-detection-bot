from tmp_train import ToxicityDataset, validate, train_loader, val_loader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

MAX_LEN = 192
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-4

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=11)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("./trained/v3.pth", map_location=device))
model = model.to(device)

accuracy, f1, precision, recall = validate()
print(f'Validation Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')
