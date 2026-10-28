import torch
from transformers import BertTokenizer, BertForTokenClassification
LOCAL_MODEL_DIR = r'.\\model\mountain_ner_bert'
# Load the fine-tuned model and tokenizer
model = BertForTokenClassification.from_pretrained(LOCAL_MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)

# Set up label mapping
id2label = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}

def predict_mountains(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process predictions
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    results = []
    for token, prediction in zip(tokens, predictions[0]):
        if token.startswith("##"):
            results[-1][0] += token[2:]
        else:
            results.append([token, id2label[prediction.item()]])
    
    # Remove special tokens
    results = [r for r in results if r[0] not in ['[CLS]', '[SEP]', '[PAD]']]
    
    return results

# Example usage
text = "I climbed Donguzorun last summer and it was an amazing experience."
result = predict_mountains(text)
print(result)