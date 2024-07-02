from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# load tokenizer and model weights
tokenizer = RobertaTokenizer.from_pretrained('./roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('./roberta_toxicity_classifier')
OUTPUT_LABEL = ["non-toxic", "toxic"]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

def predict_toxicity(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    # Assume binary classification with class 0 being non-toxic and class 1 being toxic
    prediction = probabilities.argmax(axis=1).item()
    
    return OUTPUT_LABEL[prediction]