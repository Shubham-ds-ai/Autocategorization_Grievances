import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Load BERT model and tokenizer via HuggingFace Transformers
bert = AutoModel.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')

# Assuming 'data' is your DataFrame and it contains 'cleaned_lemma' for text and 'org_code' for labels
labels, unique_labels = pd.factorize(data['org_code'])

# Create a mapping dictionary from original labels to encoded labels
label_mapping = {original_label: encoded_label for encoded_label, original_label in enumerate(unique_labels)}

# Optionally, if you want to map encoded labels back to the original labels
reverse_mapping = {v: k for k, v in label_mapping.items()}

# Add the encoded labels back to your DataFrame if needed
data['encoded_labels'] = labels

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_lemma'], data['encoded_labels'], test_size=0.2, random_state=42, stratify=data['encoded_labels'])

# Tokenize and encode sequences in the training and test set
max_length = 80  # Set max_length according to your dataset analysis

def encode_sequences(tokenizer, sequences, max_length):
    tokens = tokenizer.batch_encode_plus(
        sequences.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    return torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask'])

train_seq, train_mask = encode_sequences(tokenizer, X_train, max_length)
test_seq, test_mask = encode_sequences(tokenizer, X_test, max_length)

# Convert labels to tensors
train_y = torch.tensor(y_train.tolist())
test_y = torch.tensor(y_test.tolist())

# Data Loader structure definition
batch_size = 32  # Define a batch size, adjusted as per your GPU memory

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Define the BERT architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
        # Define additional layers
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, len(unique_labels))  # Output layer size = number of unique labels
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = BERT_Arch(bert)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Define the loss function
cross_entropy = nn.NLLLoss()

# Train and predict
best_valid_loss = float('inf')
train_losses=[]                   # empty lists to store training and validation loss of each epoch
valid_losses=[]
# Defining training and evaluation functions
def train():
  model.train()
  total_loss, total_accuracy = 0, 0

  for step,batch in enumerate(train_dataloader):                # iterate over batches
    if step % 50 == 0 and not step == 0:                        # progress update after every 100 batches.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
    batch = [r for r in batch]                                  # push the batch to gpu
    sent_id, mask, labels = batch
    model.zero_grad()                                           # clear previously calculated gradients
    preds = model(sent_id, mask)                                # get model predictions for current batch
    loss = cross_entropy(preds, labels)                         # compute loss between actual & predicted values
    total_loss = total_loss + loss.item()                       # add on to the total loss
    loss.backward()                                             # backward pass to calculate the gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # clip gradients to 1.0. It helps in preventing exploding gradient problem
    optimizer.step()                                            # update parameters
    preds=preds.detach().cpu().numpy()                          # model predictions are stored on GPU. So, push it to CPU

  avg_loss = total_loss / len(train_dataloader)                 # compute training loss of the epoch
                                                                # reshape predictions in form of (# samples, # classes)
  return avg_loss                                 # returns the loss and predictions

def evaluate():
  print("\nEvaluating...")
  model.eval()                                    # Deactivate dropout layers
  total_loss, total_accuracy = 0, 0
  for step,batch in enumerate(test_dataloader):    # Iterate over batches
    if step % 50 == 0 and not step == 0:          # Progress update every 100 batches.
                                                  # Calculate elapsed time in minutes.
                                                  # Elapsed = format_time(time.time() - t0)
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))
                                                  # Report progress
    batch = [t for t in batch]                    # Push the batch to GPU
    sent_id, mask, labels = batch
    with torch.no_grad():                         # Deactivate autograd
      preds = model(sent_id, mask)                # Model predictions
      loss = cross_entropy(preds,labels)          # Compute the validation loss between actual and predicted values
      total_loss = total_loss + loss.item()
      preds = preds.detach().cpu().numpy()
  avg_loss = total_loss / len(test_dataloader)         # compute the validation loss of the epoch
  return avg_loss

epochs=50

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss = train()                       # train model
    valid_loss = evaluate()                    # evaluate model
    if valid_loss < best_valid_loss:              # save the best model
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model_other_cat_150.pt')
    train_losses.append(train_loss)               # append training and validation loss
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')