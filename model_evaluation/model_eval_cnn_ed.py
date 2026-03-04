import torch
import torch.nn as nn
import h5py 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import os

'''This script runs an evaluation on the a CNN with encoder-decoder architecture. It calculates accuracy, precision, F1 score, and clustering metrics on the embeddings.
The model definition and data loading logic is adapted from cnn_ed_save.py, and the evaluation logic is adapted from model_eval_KD.py. '''
# Model Definition from cnn_ed_save.py
class EncoderDecoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EncoderDecoderClassifier, self).__init__()
        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=1); self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, padding=1); self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec_up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(32, 32, 3, padding=1); self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(16, 16, 3, padding=1); self.dec_bn2 = nn.BatchNorm2d(16)
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

# Utilize GPU and set paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "norm_lfw_10_shifted_CNN.pth"
data_path = 'norm_full_lfw.pt' 

# Flexible Data Loading (this logic can handle both .pt and .h5 formats)
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
else:
    raise ValueError("Unsupported file format. Please use .pt or .h5")

# Ensure correct types and shapes
X = X.float()
Y = Y.long()
if X.shape[-1] == 3:
    X = X.permute(0, 3, 1, 2)

#num_classes = 5749
num_classes = len(torch.unique(Y))
loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=False)


# Model Loading & Inference
model = EncoderDecoderClassifier(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_preds, all_trues, all_embeddings = [], [], []

print("Running evaluation...")
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        embeddings = model(x, return_embeddings=True)
        
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(y.numpy())
        all_embeddings.extend(embeddings.cpu().numpy())

all_embeddings = np.array(all_embeddings)
all_trues = np.array(all_trues)
all_preds = np.array(all_preds)


# Calculate metrics and dsiplay results
print("\n" + "="*40)
print(f"EVALUATION RESULTS ({os.path.basename(data_path)})")
print("="*40)


acc = accuracy_score(all_trues, all_preds)
prec = precision_score(all_trues, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_trues, all_preds, average='weighted', zero_division=0)
ch_score = calinski_harabasz_score(all_embeddings, all_trues)
db_index = davies_bouldin_score(all_embeddings, all_trues)

if len(all_embeddings) > 5000:
    idx = np.random.choice(len(all_embeddings), 5000, replace=False)
    sil = silhouette_score(all_embeddings[idx], all_trues[idx])
else:
    sil = silhouette_score(all_embeddings, all_trues)

print(f"{'Accuracy:':<25} {acc:.4f}")
print(f"{'Precision (W):':<25} {prec:.4f}")
print(f"{'F1 Score (W):':<25} {f1:.4f}")
print("-" * 40)
print(f"{'Calinski-Harabasz:':<25} {ch_score:.4f}")
print(f"{'Davies-Bouldin:':<25} {db_index:.4f}")
print(f"{'Silhouette Score:':<25} {sil:.4f}")
print("="*40)
