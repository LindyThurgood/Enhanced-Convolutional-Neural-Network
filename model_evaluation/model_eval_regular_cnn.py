import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import os

'''This script is designed to evaluate the performance of a regular CNN Model. It reloads the model, runs it on the full dataset, 
and computes various metrics including accuracy, f1, precision, and clustering metrics.'''

#Data Loading and Configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "raw_olivetti_plain_CNN.pth"  # Path to your saved weights
data_path = 'Olivetti.pt'       # Path to your test/eval data

print(f"Loading data from {data_path}...")

if data_path.endswith('.pt'):
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    X = data['images']
    Y = data['labels']
elif data_path.endswith('.h5'):
    with h5py.File(data_path, 'r') as hf:
        X = np.array(hf['images']) 
        Y = np.array(hf['labels'])
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

if X.ndim == 3: # [Batch, H, W] -> [Batch, 1, H, W]
    X = X.unsqueeze(1)
elif X.shape[-1] == 1 or X.shape[-1] == 3: # [Batch, H, W, C] -> [Batch, C, H, W]
    X = X.permute(0, 3, 1, 2)

X = X.float()
Y = Y.long()
#num_classes = 5749
num_classes = len(torch.unique(Y)) # Dynamically set number of classes based on data
loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=False)

#Model Definition
class StandardCNN(nn.Module):
    def __init__(self, num_classes):
        super(StandardCNN, self).__init__()
        self.layer1 = nn.Sequential(
            #lfw vs other: change input channels to 1 for grayscale should be nn.Conv2d(1, 32, kernel_size=3, padding=1)
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



#Model Loading and Evaluation 
model = StandardCNN(num_classes=num_classes).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model from {model_path}")
else:
    print(f"Warning: Model file {model_path} not found!")

model.eval()

all_preds, all_trues, all_embeddings = [], [], []

print("Running evaluation...")
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        # We explicitly call the model again for embeddings to use our return_embeddings flag
        embeddings = model(x, return_embeddings=True)
        
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(y.numpy())
        all_embeddings.extend(embeddings.cpu().numpy())

all_embeddings = np.array(all_embeddings)
all_trues = np.array(all_trues)
all_preds = np.array(all_preds)

#Metric Calculation and Display
print("\n" + "="*40)
print(f"EVALUATION RESULTS ({os.path.basename(data_path)})")
print("="*40)

# Classification Metrics
acc = accuracy_score(all_trues, all_preds)
prec = precision_score(all_trues, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)
ch_score = calinski_harabasz_score(all_embeddings, all_trues)
db_index = davies_bouldin_score(all_embeddings, all_trues)
if len(all_embeddings) > 5000:
    idx = np.random.choice(len(all_embeddings), 5000, replace=False)
    sil = silhouette_score(all_embeddings[idx], all_trues[idx])
else:
    sil = silhouette_score(all_embeddings, all_trues)

#Display results in a clear format
print(f"{'Accuracy:':<25} {acc:.4f}")
print(f"{'Precision:':<25} {prec:.4f}")
print(f"{'F1 Score:':<25} {f1:.4f}")
print("-" * 40)
print(f"{'Calinski-Harabasz:':<25} {ch_score:.4f}")
print(f"{'Davies-Bouldin:':<25} {db_index:.4f}")
print(f"{'Silhouette Score:':<25} {sil:.4f}")
print("="*40)
