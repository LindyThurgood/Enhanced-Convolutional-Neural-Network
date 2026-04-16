import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

'''This script utilizes a pre-trained teacher model to train a student model on the full LFW dataset using knowledge distillation. 
This code was adapted from code that Maryam Bagharian gave me for a single teacher, and I implimented a hard loss function to aide in creating a more robust student model.'''

# Set up and data loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = 'olivetti_denoised_zscore_aug2.pt'
teacher_model_path = 'Olivetti_aug_CNN_ED.pth'
model_path = "kd_ed_Olivetti_T2A05.pth"

T = 2       
alpha = 0.5   
epochs = 150 
initial_lr = 0.001

data = torch.load(data_path, weights_only=True)
X, Y = data['images'].float(), data['labels'].long()
num_classes = len(torch.unique(Y))

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=27)
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

# Initialize Models, Optimizer, and Scheduler
teacher = TeacherModel(num_classes).to(device)
teacher.load_state_dict(torch.load(teacher_model_path, weights_only=True))
teacher.eval()

student = StudentModel(num_classes).to(device)
optimizer = optim.Adam(student.parameters(), lr=initial_lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- 4. DISTILLATION TRAINING LOOP ---
print("Starting Distillation with Enhanced Separation Logic...")
best_val_acc = 0.0

for epoch in range(epochs):
    student.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Get Student Embeddings and Logits
        student_emb = student(inputs, return_embeddings=True)
        student_logits = student.classifier(student_emb)
        
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # 1. Soft Loss 
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T * T)
        
        # 2. Hard Loss 
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 3. Combined Loss
        loss = (alpha * soft_loss) + ((1 - alpha) * hard_loss)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validation
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

print(f"\nDistillation Complete. Best Val Accuracy: {best_val_acc:.4f}")
