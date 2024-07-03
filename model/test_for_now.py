from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# load tokenizer and model weights
tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

print(model)

# prepare the input
batch = tokenizer.encode('you are not that smart, but ok', return_tensors='pt')

# inference
result = model(batch)


# convert logits to probabilities
probabilities = torch.nn.functional.softmax(result.logits, dim=1)

print(probabilities)

# round the probabilities
probabilities = torch.round(probabilities)

# print the result
print(probabilities)

