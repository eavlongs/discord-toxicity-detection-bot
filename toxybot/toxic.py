from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# load tokenizer and model weights
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('./model')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_LABEL = ["non-toxic", "toxic"]

def predict_toxicity(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    prediction = probabilities.argmax(axis=1).item()

    return OUTPUT_LABEL[prediction]