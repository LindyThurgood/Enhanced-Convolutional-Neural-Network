import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import h5py
'''This script trains a standard CNN, saves the model with the best validation accuracy, extracts feature vectors from the same model and saves them to an HDF5 file.'''

# hardwarre and data
h5_name = 'raw_subset_10_plain_embeddings.h5'
model_path = 'raw_subset_10_plain_CNN.pth'
data_path= 'lfw_subset_10.pt'
batch_sz = 16 
max_epochs = 200
best_val_acc = 0.0  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data from:", data_path)
data = torch.load(data_path, weights_only=True)
X = data['images'].float() 
Y = data['labels'].long()
#num_classes=5749
num_classes = len(torch.unique(Y))
print(f"Data Loaded. Shape: {X.shape}, Classes: {num_classes}")
print(f"Final Shape of X for training: {X.shape}")
if not isinstance(X, torch.Tensor):
    X = torch.tensor(X)

# Check if X is in the correct shape [Batch, Channels, Height, Width]
if X.ndim == 3: # If [Batch, H, W]
    X = X.unsqueeze(1)
elif X.shape[-1] == 1 or X.shape[-1] == 3: # If [Batch, H, W, C]
    X = X.permute(0, 3, 1, 2)

# debugging prints
print(f"Final Shape of X for training: {X.shape}") 
print(f"Data Loaded. Classes: {num_classes}")
print(f"Min Label: {Y.min()}, Max Label: {Y.max()}")
print(f"Num Classes: {num_classes}")

# Split and Create Loaders
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=27
)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_sz, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_sz, shuffle=False)

# Model Definition 
class StandardCNN(nn.Module):
    def __init__(self, num_classes):
        super(StandardCNN, self).__init__()
        
        
        self.layer1 = nn.Sequential(
            # LFW vs other: change input channels to 1 for grayscale should be nn.Conv2d(1, 32, kernel_size=3, padding=1)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 32 -> 64 filters
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
        # Feature Vector Layer
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

#Training Loop
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
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"  --> [SAVED] New Best Accuracy: {best_val_acc:.4f}")

# 6. Extraction
print(f"\nReloading Best Weights for Feature Vector Extraction...")
model.load_state_dict(torch.load(model_path))
model.eval()

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

print(f"Process complete. Feature vectors saved to {h5_name}")
