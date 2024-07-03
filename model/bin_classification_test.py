import torch
from transformers import RobertaTokenizer, RobertaModel
from tmp_train import ToxicityClassifer
import torch.nn.functional as F
# Model definition
model = ToxicityClassifer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model_path = './trained/v9.pth'  # Adjust the path if necessary
model.load_state_dict(torch.load(model_path, map_location=device))

print(model)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
MAX_LEN = 196
def get_prediction(comment):
    # Tokenize the comment
    inputs = tokenizer.encode_plus(
        comment,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
    )

    ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
    mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([inputs['token_type_ids']], dtype=torch.long).to(device)
    # ids = data['ids'].to(device, dtype = torch.long)
    # mask = data['mask'].to(device, dtype = torch.long)
    # token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
    # targets = data['targets'].to(device, dtype = torch.long)
    # outputs = model(ids, mask, token_type_ids)
    # Make prediction
    # model.eval()
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
        # Apply softmax to convert logits to probabilities
        prediction = torch.sigmoid(outputs).item()

    return prediction

print("predicted:", get_prediction("hello nice to meet you"))