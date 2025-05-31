#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import yaml

import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import pandas as pd
from nemo.collections.asr.models import EncDecRNNTBPEModel
from Baseline_utils import CEMModel

with open("./train_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader) 
    
train_data_file_path = config["train_data_file_path"] #  librispeech_train
test_data_file_path = config["test_data_file_path"] # librispeech_test_baseline
df_path = config["df_path"]
checkpoint_save_path = config["checkpoint_save_path"]
checkpoint_path = config["checkpoint_path"]
loss = config["loss"]

class CustomLoss(nn.Module): # shrinkage loss class. 
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, a=2.0 , c=0.5): # a means 'gamma', c means 'K' 
        l2criterion = nn.MSELoss(reduction='mean')
        l2loss = l2criterion(output, target)
        l1criterion = nn.L1Loss(reduction='mean') 
        l1loss = l1criterion(output, target)
        shrinkage = (1 + (a*(c-l1loss)).exp()).reciprocal()
        loss = shrinkage * l2loss * output.exp() 
        loss = torch.mean(loss) 
        
        return loss

from torch.utils.data import Dataset, DataLoader
import pickle
class ConfidenceDataset(Dataset):
    def __init__(self, file_path, return_tokens=False):
        """
        Args:
            file_path (str): Path to the pickle file containing data.
            return_tokens (bool): Whether to return token numbers.
        """
        self.file_path = file_path
        self.return_tokens = return_tokens
        self.data = list(self._load_data())  # Load data into memory

    def _load_data(self):
        """Generator that yields (features, labels, [tokens]) from a pickle file."""
        with open(self.file_path, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)  # Load the dictionary

                    if 'acoustic_features' not in data or not data['acoustic_features']:
                        continue  # Skip this entry

                    features = torch.stack(data['acoustic_features'])  # Stack tensors
                    
                    if 'emission_time_label' not in data or not data['emission_time_label']:
                        continue  # Skip this entry

                    # Extract labels (last element of each tuple)
                    labels = torch.tensor([t[-1] for t in data['emission_time_label']], dtype=torch.float32)
                    
                    if self.return_tokens:
                        tokens = torch.tensor([t[1] for t in data['emission_time_label']], dtype=torch.long)
                        yield features, labels, tokens
                    else:
                        yield features, labels
                
                except EOFError:
                    break
                # except Exception as e:
                #     print(f"Error loading data: {e}, Data: {data}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Returns (features, labels) or (features, labels, tokens)

def collate_fn(batch):
    if len(batch[0]) == 3:
        features, labels, tokens = zip(*batch)
    else:
        features, labels = zip(*batch)
        tokens = None

    lengths = torch.tensor([len(seq) for seq in features])
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    return (padded_features, padded_labels, lengths, tokens) if tokens is not None else (padded_features, padded_labels, lengths)

def get_dataloader(file_path, batch_size=32, shuffle=True, return_tokens=False):
    dataset = ConfidenceDataset(file_path, return_tokens=return_tokens)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    


# Create DataLoaders
train_dataloader = get_dataloader(train_data_file_path, batch_size=512, return_tokens=False)
test_dataloader = get_dataloader(test_data_file_path, batch_size=64, shuffle=False, return_tokens=True)

device = "cuda"
if loss == 'mae':
    criterion_confid = nn.L1Loss(reduction='mean').to(device)
if loss == 'shrinkage':
    criterion_confid = CustomLoss().to(device)
# In[ ]:


# def evaluate(model, dataloader, tokenizer, device='cuda'):
def evaluate(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0.0
    y_true_list, y_score_list = [], []
    
    with torch.no_grad():
        for batch_features, batch_labels, lengths, _ in dataloader:
            batch_features, batch_labels, lengths = (
                batch_features.to(device),
                batch_labels.to(device),
                lengths.to(device),
            )
            predictions = model(batch_features, lengths)
            seq_len = batch_labels.shape[1]
            predictions = predictions[:, :seq_len]
            mask = batch_labels != -1
            valid_preds = predictions[mask]
            valid_targets = batch_labels[mask]
            loss = criterion_confid(valid_preds, valid_targets)
            total_loss += loss.item() 

    return {
        "val_loss": total_loss / len(dataloader),
    }

def train(model, train_dataloader, test_dataloader, epochs=10, lr=1e-3, device='cuda'):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    metrics_list = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_features, batch_labels, lengths in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features, batch_labels, lengths = (
                batch_features.to(device),
                batch_labels.to(device),
                lengths.to(device),
            )
            
            predictions = model(batch_features, lengths)
            seq_len = batch_labels.shape[1]

            predictions = predictions[:, :seq_len]
            
            mask = batch_labels != -1
            valid_preds = predictions[mask]
            valid_targets = batch_labels[mask]
            
            loss = criterion_confid(valid_preds, valid_targets)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() 
        
        train_loss = epoch_loss / len(train_dataloader)
        # val_metrics = evaluate(model, test_dataloader, tokenizer, device)
        val_metrics = evaluate(model, test_dataloader, device)
        val_loss = val_metrics["val_loss"]
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_save_path)
            print(f"Model saved at {checkpoint_save_path}")
        
        # Append metrics to list
        metrics_list.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,

        })
    
    # Save metrics to CSV
    df = pd.DataFrame(metrics_list)
    df.to_csv(df_path, index=False)
    print(f"Training metrics saved to {df_path}")


# Load checkpoint if available
cem_model = CEMModel(hidden_dim=512, num_layers=2)

try:
    checkpoint = torch.load(checkpoint_path)
    cem_model.load_state_dict(checkpoint)
    print("Checkpoint loaded. Resuming training...")
except FileNotFoundError:
    print("No checkpoint found. Starting fresh training.")

train(cem_model, train_dataloader, test_dataloader, epochs=50, lr=0.001, device='cuda:3' if torch.cuda.is_available() else 'cpu')

