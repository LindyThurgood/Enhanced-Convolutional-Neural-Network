import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)
import numpy as np
import h5py
from norm_abide import augment_connectivity_matrices
from Norm4D import augment_images
from new_aug import perform_torchvision_augmentation
import torchvision.transforms as T

'''
This script trains a standard CNN, saves the best model, and calculates 
classification (F1) and clustering (CHI, DBI, Silhouette) metrics on validation data.
It also saves the feature vectors of the best model and allows for training class 
augmentation in 3 different forms: connectivity matrices, added noise, and image augmentation. 
'''

# Hardware and Data Configuration
h5_name = 'DNi_Olivetti_Plain_CNN_embeddings.h5'
model_path = 'DNi_Olivetti_Plain_CNN.pth'
data_path = 'DN_Olivetti.pt'
batch_sz = 16 
max_epochs = 200
best_val_acc = 0.0  
# Augmentation set up. connectivity indicates that the data is connectivity matrices and is_images indicates that the data should be augmented like images (rotations, brightness change...)
augment = True 
augmentation_factor = 6
connectivity = False
is_images= False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
print("Loading data from:", data_path)
data = torch.load(data_path, weights_only=True)
X = data['images'].float() 
Y = data['labels'].long()
num_classes = len(torch.unique(Y))

if not isinstance(X, torch.Tensor):
    X = torch.tensor(X)

# Ensure correct shape [Batch, Channels, Height, Width]
if X.ndim == 3:
    X = X.unsqueeze(1)
elif X.shape[-1] == 1 or X.shape[-1] == 3:
    X = X.permute(0, 3, 1, 2)

print(f"Data Loaded. Shape: {X.shape}, Classes: {num_classes}")

# Split Data
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.3, shuffle=True
) #random_state=27

# Augmentation
if augment:
    if connectivity:
        X_train_np = X_train.squeeze(1).cpu().numpy()
        Y_train_np = Y_train.cpu().numpy()
        X_train_np, Y_train_np = augment_connectivity_matrices(X_train_np, Y_train_np, augmentation_factor)
        X_train = torch.from_numpy(X_train_np).float().unsqueeze(1)
        Y_train = torch.from_numpy(Y_train_np).long()
        print(f"Connectivity Augmentation complete. New training size: {X_train.shape[0]}")
    elif is_images: 
        X_train, Y_train = perform_torchvision_augmentation(X_train, Y_train, augmentation_factor)
        if X_train.ndim == 3:
            X_train = X_train.unsqueeze(1)
        print(f"Image Augmentation complete. New size: {X_train.shape[0]}")
    else:
        X_train_np = X_train.squeeze(1).cpu().numpy()
        Y_train_np = Y_train.cpu().numpy()
        X_train_np, Y_train_np = augment_images(X_train_np, Y_train_np, augmentation_factor) 
        X_train = torch.from_numpy(X_train_np).float().unsqueeze(1)
        Y_train = torch.from_numpy(Y_train_np).long()
        print(f"Image Augmentation complete. New training size: {X_train.shape[0]}")
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_sz, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_sz, shuffle=False)

# Model Definition 
class StandardCNN(nn.Module):
    def __init__(self, num_classes):
        super(StandardCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc1 = nn.Linear(128, 256) 
        self.dropout = nn.Dropout(0.4)
        self.fc_final = nn.Linear(256, num_classes)
        
    def forward(self, x, return_embeddings=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).view(x.size(0), -1)
        embedding = torch.relu(self.fc1(x))
        if return_embeddings: 
            return embedding
        return self.fc_final(self.dropout(embedding))

model = StandardCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
print("\nStarting Training...")
for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Simple Validation for checkpointing
    model.eval()
    all_preds, all_truth = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_truth.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(all_truth, all_preds)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"  --> [SAVED] New Best Accuracy: {best_val_acc:.4f}")

# Final Metric Calculation & Extraction
print(f"\nReloading Best Weights for Metric Calculation...")
model.load_state_dict(torch.load(model_path))
model.eval()

# Metrics containers for the validation set
val_embeddings, val_labels, val_preds = [], [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        # Get predictions for F1
        logits = model(inputs)
        _, preds = torch.max(logits, 1)
        # Get embeddings for clustering metrics
        emb_batch = model(inputs, return_embeddings=True)
        
        val_embeddings.append(emb_batch.cpu().numpy())
        val_labels.append(labels.numpy())
        val_preds.append(preds.cpu().numpy())

# Convert to arrays
val_embeddings_arr = np.concatenate(val_embeddings, axis=0)
val_labels_arr = np.concatenate(val_labels, axis=0)
val_preds_arr = np.concatenate(val_preds, axis=0)

# Calculate Scores
final_f1 = f1_score(val_labels_arr, val_preds_arr, average='weighted')
final_acc = accuracy_score(val_labels_arr, val_preds_arr)
sil = silhouette_score(val_embeddings_arr, val_labels_arr)
chi = calinski_harabasz_score(val_embeddings_arr, val_labels_arr)
dbi = davies_bouldin_score(val_embeddings_arr, val_labels_arr)

# Print Results
print("\n" + "="*40)
print("       FINAL VALIDATION RESULTS")
print("="*40)
print(f"Accuracy:         {final_acc:.4f}")
print(f"F1 Score (Wtd):   {final_f1:.4f}")
print("-" * 40)
print(f"Silhouette Score: {sil:.4f}  (Ideal: 1.0)")
print(f"CHI Score:        {chi:.4f} (Higher is better)")
print(f"DBI Score:        {dbi:.4f}  (Lower is better)")
print("="*40)

# 8. Save Full Dataset Embeddings (Original requirement)
print(f"\nExtracting all embeddings for HDF5 storage...")
full_loader = DataLoader(TensorDataset(X, Y), batch_size=batch_sz)
all_embeddings, all_labels = [], []

with torch.no_grad():
    for inputs, labels in full_loader:
        inputs = inputs.to(device)
        emb_batch = model(inputs, return_embeddings=True)
        all_embeddings.append(emb_batch.cpu().numpy())
        all_labels.append(labels.numpy())

embeddings_array = np.concatenate(all_embeddings, axis=0)
labels_array = np.concatenate(all_labels, axis=0)

with h5py.File(h5_name, 'w') as hf: 
    hf.create_dataset('feature_vectors', data=embeddings_array)
    hf.create_dataset('labels', data=labels_array)

print(f"Process complete. Data saved to {h5_name}")
