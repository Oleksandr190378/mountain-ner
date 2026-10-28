import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
from tqdm import tqdm


def prepare_data(sentence, mountain):
    tokens = tokenizer.tokenize(sentence)
    labels = ['O'] * len(tokens)
    mountain_tokens = tokenizer.tokenize(mountain)
    
    for i in range(len(tokens)):
        if tokens[i:i+len(mountain_tokens)] == mountain_tokens:
            labels[i] = 'B-MOUNTAIN'
            for j in range(1, len(mountain_tokens)):
                if i+j < len(labels):
                    labels[i+j] = 'I-MOUNTAIN'
    
    return tokens, labels



df = pd.read_csv(r'./data/mountain_sentences.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)
all_tokens = []
all_labels = []

for _, row in df.iterrows():
    tokens, labels = prepare_data(row['sentence'], row['mountain'])
    all_tokens.append(tokens)
    all_labels.append(labels)

label_map = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}
input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in all_tokens]
label_ids = [[label_map[label] for label in labels] for labels in all_labels]

max_len = max(len(seq) for seq in input_ids)
input_ids = [seq + [0] * (max_len - len(seq)) for seq in input_ids]
label_ids = [seq + [0] * (max_len - len(seq)) for seq in label_ids]

input_ids = torch.tensor(input_ids)
label_ids = torch.tensor(label_ids)

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, label_ids, test_size=0.2)

train_data = TensorDataset(train_inputs, train_labels)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(val_inputs, val_labels)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 10


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
        input_ids, labels = batch
        attention_mask = (input_ids != 0).long()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Average training loss: {avg_train_loss:.4f}')


# Validation
model.eval()
val_loss = 0

for batch in val_dataloader:
    input_ids, labels = batch
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    
    val_loss += outputs.loss.item()

avg_val_loss = val_loss / len(val_dataloader)
print(f'Validation loss: {avg_val_loss:.4f}')    


LOCAL_MODEL_DIR = r'.\model\mountain_ner_bert'
model.save_pretrained(LOCAL_MODEL_DIR)
tokenizer.save_pretrained(LOCAL_MODEL_DIR)


# Analyze model performance
print("\nModel performance analysis:")

model.eval()
with torch.no_grad():
    outputs = model(val_inputs)

preds = torch.argmax(outputs.logits, dim=2)

# Flatten predictions and labels, removing padding tokens
true_labels = val_labels.view(-1).numpy()
pred_labels = preds.view(-1).numpy()

valid_indices = true_labels != -100
true_labels = true_labels[valid_indices]
pred_labels = pred_labels[valid_indices]

print(classification_report(true_labels, pred_labels, target_names=['O', 'B-MOUNTAIN', 'I-MOUNTAIN'], zero_division=0))
