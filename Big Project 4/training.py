import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import ast
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def parse_evidence(evidence_str):
    evidence_list = ast.literal_eval(evidence_str)
    evidence_text = ' '.join([item[2] for sublist in evidence_list for item in sublist if item[2] is not None])
    return evidence_text

# Read the CSV file
data = pd.read_csv('data.csv')

# Preprocess the data
data['evidence'] = data['evidence'].apply(lambda x: parse_evidence(x))
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=2/3, random_state=42)


# Tokenize and encode the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_data['claim'].tolist(), train_data['evidence'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer(val_data['claim'].tolist(), val_data['evidence'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_data['claim'].tolist(), test_data['evidence'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')

# Create label mappings
label_mapping = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
train_labels = train_data['label'].apply(lambda x: label_mapping[x]).tolist()
val_labels = val_data['label'].apply(lambda x: label_mapping[x]).tolist()
test_labels = test_data['label'].apply(lambda x: label_mapping[x]).tolist()


# Create PyTorch tensors and data loaders
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load a pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model
epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    print(f"Epoch {epoch + 1} Train Loss: {train_loss / len(train_loader)}")

# Validate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    print(f"Epoch {epoch + 1} Validation Loss: {val_loss / len(val_loader)}")

# Save the model
torch.save(model.state_dict(), 'fact_verification_bert_model.pt')

# Evaluate the model on the test set
model.eval()
test_loss = 0
test_predictions = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        test_loss += loss.item()
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        test_predictions.extend(np.argmax(logits, axis=1))
        test_labels.extend(label_ids)

test_accuracy = np.sum(np.array(test_predictions) == np.array(test_labels)) / len(test_labels)
print(f"Test Accuracy: {test_accuracy}")
print(classification_report(test_labels, test_predictions, target_names=label_mapping.keys()))
print(confusion_matrix(test_labels, test_predictions))



def evaluate_transformed_data(file_path, transformation_name):
    transformed_data = pd.read_csv(file_path)
    transformed_data['evidence'] = transformed_data['evidence'].apply(lambda x: parse_evidence(x))

    # Split the transformed data into training, validation, and test sets (70%, 10%, 20%)
    transformed_temp_data = train_test_split(transformed_data, test_size=0.3, random_state=42)
    transformed_test_data = train_test_split(transformed_temp_data, test_size=2/3, random_state=42)

    transformed_encodings = tokenizer(transformed_test_data['claim'].tolist(), transformed_test_data['evidence'].tolist(), 
    padding=True, truncation=True, max_length=128, return_tensors='pt')
    transformed_labels = transformed_test_data['label'].apply(lambda x: label_mapping[x]).tolist()

    transformed_dataset = TensorDataset(transformed_encodings['input_ids'], transformed_encodings['attention_mask'], torch.tensor(transformed_labels))
    transformed_loader = DataLoader(transformed_dataset, batch_size=16, shuffle=False)

    model.eval()
    transformed_test_loss = 0
    transformed_test_predictions = []
    transformed_test_labels = []
    with torch.no_grad():
        for batch in transformed_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            transformed_test_loss += loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            transformed_test_predictions.extend(np.argmax(logits, axis=1))
            transformed_test_labels.extend(label_ids)

    transformed_test_accuracy = np.sum(np.array(transformed_test_predictions) == np.array(transformed_test_labels)) / len(transformed_test_labels)
    print(f"{transformation_name} Test Accuracy: {transformed_test_accuracy}")
    print(classification_report(transformed_test_labels, transformed_test_predictions, target_names=label_mapping.keys()))
    print(confusion_matrix(transformed_test_labels, transformed_test_predictions))

evaluate_transformed_data("output_synonyms.csv", "Synonyms")
evaluate_transformed_data("output_antonyms.csv", "Antonyms")
evaluate_transformed_data("output_negation.csv", "Negation")
evaluate_transformed_data("output_swap_adjacent_words.csv", "Swap Adjacent Words")
evaluate_transformed_data("output_insert_redundant_phrase.csv", "Insert Redundant Phrase")
evaluate_transformed_data("output_typo.csv", "Typo")

