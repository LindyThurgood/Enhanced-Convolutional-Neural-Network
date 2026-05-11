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
from new_aug import perform_torchvision_augmentation
import torchvision.transforms as T
from Norm4D import augment_images


'''
This script trains an encoder-decoder CNN, saves the best model, 
and calculates classification and clustering metrics on validation data.
'''

# File names and paths
h5_name = 'DNi_Olivetti_EncoderDecoder_embeddings.h5'
model_path = 'DNi_Olivetti_EncoderDecoder.pth'
data_path = 'DN_Olivetti.pt'
# Augmentation set up. connectivity indicates that the data is connectivity matrices and is_images indicates that the data should be augmented like images (rotations, brightness change...)
augment = True  
augmentation_factor = 6
connectivity = False
is_images= False

batch_sz = 16 
max_epochs = 200
best_val_acc = 0.3

# Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
print("Loading data from .pt file...")
data = torch.load(data_path, weights_only=True)
X = data['images'].float() 
Y = data['labels'].long() 

# Ensure correct shape [Batch, Channels, Height, Width]
if X.ndim == 3:
    X = X.unsqueeze(1)
elif X.shape[-1] == 1 or X.shape[-1] == 3:
    X = X.permute(0, 3, 1, 2)

num_classes = len(torch.unique(Y))
print(f"Data Loaded. Shape: {X.shape}, Classes: {num_classes}")

# Split Data
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.3, shuffle=True, random_state=27
)

# Augment the training data 
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
class EncoderDecoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EncoderDecoderClassifier, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 16, 3, padding=1); self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, padding=1); self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_pool2 = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        # Decoder
        self.dec_up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(32, 32, 3, padding=1); self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(16, 16, 3, padding=1); self.dec_bn2 = nn.BatchNorm2d(16)
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc1 = nn.Linear(16, 32) 
        self.dropout = nn.Dropout(0.3)
        self.fc_final = nn.Linear(32, num_classes)
        
    def forward(self, x, return_embeddings=False):
        x = torch.relu(self.enc_bn1(self.enc_conv1(x))); x = self.enc_pool1(x)
        x = torch.relu(self.enc_bn2(self.enc_conv2(x))); x = self.enc_pool2(x)
        x = self.bottleneck(x)
        x = torch.relu(self.dec_up1(x)); x = torch.relu(self.dec_bn1(self.dec_conv1(x)))
        x = torch.relu(self.dec_up2(x)); x = torch.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.gap(x).view(x.size(0), -1)
        embedding = self.fc1(x) 
        if return_embeddings: return embedding
        return self.fc_final(self.dropout(embedding))

model = EncoderDecoderClassifier(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
print("\nStarting Training:")
for epoch in range(max_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
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
    print(f"Epoch {epoch+1}/{max_epochs} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"  --> [SAVED] New Best Accuracy!: {best_val_acc:.4f}")

# Extraction & Metric Calculation
print(f"\nReloading Best Weights (Acc: {best_val_acc:.4f}) for Metric Calculation...")
model.load_state_dict(torch.load(model_path))
model.eval()

# Metrics containers for the validation set specifically
val_embeddings, val_labels, val_preds = [], [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        # Predictions for F1
        logits = model(inputs)
        _, preds = torch.max(logits, 1)
        # Embeddings for Clustering metrics
        emb_batch = model(inputs, return_embeddings=True)
        
        val_embeddings.append(emb_batch.cpu().numpy())
        val_labels.append(labels.numpy())
        val_preds.append(preds.cpu().numpy())

# Convert to arrays
val_embeddings_arr = np.concatenate(val_embeddings, axis=0)
val_labels_arr = np.concatenate(val_labels, axis=0)
val_preds_arr = np.concatenate(val_preds, axis=0)

# Calculate final Metrics
final_f1 = f1_score(val_labels_arr, val_preds_arr, average='weighted')
sil = silhouette_score(val_embeddings_arr, val_labels_arr)
chi = calinski_harabasz_score(val_embeddings_arr, val_labels_arr)
dbi = davies_bouldin_score(val_embeddings_arr, val_labels_arr)

print("\n" + "="*40)
print("       FINAL VALIDATION RESULTS")
print("="*40)
print(f"Best Accuracy:    {best_val_acc:.4f}")
print(f"F1 Score (Wtd):   {final_f1:.4f}")
print("-" * 40)
print(f"Silhouette Score: {sil:.4f}")
print(f"CHI Score:        {chi:.4f}")
print(f"DBI Score:        {dbi:.4f}")
print("="*40)

# Extract and Save all embeddings to HDF5
print(f"\nExtracting all embeddings for full dataset storage...")
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
    hf.create_dataset('embeddings', data=embeddings_array)
    hf.create_dataset('labels', data=labels_array)

print(f"Process complete. Embeddings saved to {h5_name}")
