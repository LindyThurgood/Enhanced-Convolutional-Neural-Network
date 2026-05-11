import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)
from norm_abide import augment_connectivity_matrices
from Norm4D import augment_images
from new_aug import perform_torchvision_augmentation
import torchvision.transforms as T

'''This script utilizes a pre-trained teacher model to train a student model using knowledge distillation.'''

# Set up and data loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = 'DN_Olivetti.pt'
teacher_model_path = 'DNi_Olivetti_EncoderDecoder.pth'
model_path = "KD_DNi_Olivetti_EncoderDecoder.pth"

# Augmentation set up. connectivity indicates that the data is connectivity matrices and is_images indicates that the data should be augmented like images (rotations, brightness change...)
augment = True  
augmentation_factor = 6
connectivity = False
is_images= False

T = 2        
alpha = 0.5    
epochs = 150 
initial_lr = 0.001

data = torch.load(data_path, weights_only=True)
X, Y = data['images'].float(), data['labels'].long()
num_classes = len(torch.unique(Y))
if X.ndim == 3:
    X = X.unsqueeze(1)
elif X.shape[-1] == 1 or X.shape[-1] == 3:
    X = X.permute(0, 3, 1, 2)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=27)

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

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=16)

# Model Definitions

class TeacherModel(nn.Module):
    def __init__(self, num_classes):
        super(TeacherModel, self).__init__()
        self.enc_conv1 = nn.Conv2d(1, 16, 3, padding=1); self.enc_bn1 = nn.BatchNorm2d(16)
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
        self.fc_final = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.enc_bn1(self.enc_conv1(x))); x = self.enc_pool1(x)
        x = torch.relu(self.enc_bn2(self.enc_conv2(x))); x = self.enc_pool2(x)
        x = self.bottleneck(x)
        x = torch.relu(self.dec_up1(x)); x = torch.relu(self.dec_bn1(self.dec_conv1(x)))
        x = torch.relu(self.dec_up2(x)); x = torch.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc_final(x)

class StudentModel(nn.Module): 
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(256, 128)
        self.bn_emb = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        emb = self.bn_emb(self.embedding(x))
        if return_embeddings:
            return emb
        return self.classifier(emb)

# Initialize Models
teacher = TeacherModel(num_classes).to(device)
teacher.load_state_dict(torch.load(teacher_model_path, weights_only=True))
teacher.eval()

student = StudentModel(num_classes).to(device)
optimizer = optim.Adam(student.parameters(), lr=initial_lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Distilation Training Loop
print("Starting Distillation Training...")
best_val_acc = 0.0

for epoch in range(epochs):
    student.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        student_emb = student(inputs, return_embeddings=True)
        student_logits = student.classifier(student_emb)
        
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T * T)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        loss = (alpha * soft_loss) + ((1 - alpha) * hard_loss)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Simple Validation for Checkpointing
    student.eval()
    all_preds = []
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            outputs = student(val_inputs.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
    
    val_acc = accuracy_score(Y_val.numpy(), all_preds)
    
    if (epoch + 1) % 10 == 0 or val_acc > best_val_acc:
        print(f"Epoch {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(student.state_dict(), model_path)
        print(f"  --> [SAVED] New Best Accuracy: {best_val_acc:.4f}")

# Final Metric Calculation
print(f"\nDistillation Complete. Reloading Best Student for Final Metrics...")
student.load_state_dict(torch.load(model_path))
student.eval()

val_embeddings, val_labels, val_preds = [], [], []

with torch.no_grad():
    for val_inputs, val_targets in val_loader:
        val_inputs = val_inputs.to(device)
        # Get Predictions
        logits = student(val_inputs)
        preds = logits.argmax(1)
        # Get Latent Embeddings
        embs = student(val_inputs, return_embeddings=True)
        
        val_embeddings.append(embs.cpu().numpy())
        val_labels.append(val_targets.numpy())
        val_preds.append(preds.cpu().numpy())

# Convert to single arrays
val_embeddings_arr = np.concatenate(val_embeddings, axis=0)
val_labels_arr = np.concatenate(val_labels, axis=0)
val_preds_arr = np.concatenate(val_preds, axis=0)

# Calculate Scores
final_f1 = f1_score(val_labels_arr, val_preds_arr, average='weighted')
sil = silhouette_score(val_embeddings_arr, val_labels_arr)
chi = calinski_harabasz_score(val_embeddings_arr, val_labels_arr)
dbi = davies_bouldin_score(val_embeddings_arr, val_labels_arr)

print("\n" + "="*40)
print("       STUDENT MODEL FINAL METRICS")
print("="*40)
print(f"Final Accuracy:   {best_val_acc:.4f}")
print(f"F1 Score (Wtd):   {final_f1:.4f}")
print("-" * 40)
print(f"Silhouette Score: {sil:.4f}")
print(f"CHI Score:        {chi:.4f}")
print(f"DBI Score:        {dbi:.4f}")
print("="*40)
