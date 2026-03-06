import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import h5py


'''This script trains an encoder-decoder CNN currently set up to run on the dimensions of the LFW dataset. 
It creates the CNN, saves the best model, and extracts embeddings to an HDF5 file.'''

#File names and paths
h5_name = 'norm_lfw_10_aug_3_shifted_CNN.h5'
model_path = 'norm_lfw_10_aug_3_shifted_CNN.pth'
data_path= 'lfw_full_norm_aug3_10_shifted.pt'
# Hardware Setup - utilize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load the PyTorch Data 
print("Loading data from .pt file...")
data = torch.load(data_path, weights_only=True)
X = data['images'].float() #adjust to the key used in your .pt file
Y = data['labels'].long() 
num_classes = 5749
'''find the correct number of classes from the labels can be hard coded as well 
for a shifted version of the data, the number of classes need to be hard coded to the correct number of classes in the shifted data
the range must always be from 0 to num_classes-1'''
#num_classes = len(torch.unique(Y))
X = X.permute(0, 3, 1, 2)
print(f"Data Loaded. Shape: {X.shape}, Classes: {num_classes}")
print(f"Max label found in data: {Y.max()}")
print(f"Expected classes: {num_classes}")

#Split and Create Loaders
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=27
)

batch_sz = 16 
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_sz, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_sz, shuffle=False)

# Model Definition with encoder-decoder architecture
class EncoderDecoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EncoderDecoderClassifier, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=1); self.enc_bn1 = nn.BatchNorm2d(16)
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

#Training Loop
max_epochs = 200
best_val_acc = .3 #save criteria for model


print("\nStarting Training:")
for epoch in range(max_epochs):
    model.train()
    #Training
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
        print(f"  --> Model Saved New Best Accuracy!: {best_val_acc:.4f}")

# 6. Extraction
print(f"\nReloading Best Weights (Acc: {best_val_acc:.4f}) for Embedding Extraction...")
model.load_state_dict(torch.load(model_path))
model.eval()

# Extract embeddings for the entire dataset (train + val)
full_loader = DataLoader(TensorDataset(X, Y), batch_size=batch_sz)
all_embeddings, all_labels = [], []

with torch.no_grad():
    for inputs, labels in full_loader:
        inputs = inputs.to(device)
        emb_batch = model(inputs, return_embeddings=True)
        all_embeddings.append(emb_batch.cpu().numpy())
        all_labels.append(labels.numpy())

# Finalize and Save
embeddings_array = np.concatenate(all_embeddings, axis=0)
labels_array = np.concatenate(all_labels, axis=0)


with h5py.File(h5_name, 'w') as hf: 
    hf.create_dataset('embeddings', data=embeddings_array)
    hf.create_dataset('labels', data=labels_array)

print(f"Embeddings saved to {h5_name}")
