import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

'''This script utilizes two pre-trained teacher models (T1 and T2) to train a student model on the full LFW dataset using knowledge distillation. 
It also utilizes selective distillation, where the student learns from the teacher that is an expert on the specific sample based on the dataset split. 
The student model is designed to be more complex than the teachers to effectively learn from both. It is currently set up to accept two teachers with endcoder decoder architectures. 
This code was adapted from code that Maryam Bagharian gave me for a single teacher, and I modified it to handle two teachers and the selective distillation logic.'''

# Device Configuration and Data Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_INDEX = 5591  # T1: 0-5590, T2: 5591-5748
BATCH_SIZE = 128
T = 0.5         
alpha = 0.05  
teacher1_path = "norm_lfw_9_cnn.pth"
teacher2_path = "norm_lfw_10_aug_3_shifted_CNN.pth"       
data_path = 'norm_full_lfw.pt'
student_path = "lfw_dual_teacher_student_T05A05.pth"

# Load Data
data = torch.load(data_path, map_location='cpu', weights_only=False)
X = data['images']
Y = data['labels']

if isinstance(X, np.ndarray):
    X = torch.from_numpy(X).float().transpose(3, 1) 
    Y = torch.from_numpy(Y).long()
else:
    X = X.float().permute(0, 3, 1, 2) if X.shape[1] != 3 else X.float()
    Y = Y.long()

num_classes = len(torch.unique(Y))

X_train, X_val, Y_train, Y_val = train_test_split(
    X.numpy(), Y.numpy(), test_size=0.2, shuffle=True, random_state=42
)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
    batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
    batch_size=BATCH_SIZE, shuffle=False
)


# Model Definitions
class TeacherNet(nn.Module):
    def __init__(self, num_classes):
        super(TeacherNet, self).__init__()
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
        
    def forward(self, x):
        x = torch.relu(self.enc_bn1(self.enc_conv1(x))); x = self.enc_pool1(x)
        x = torch.relu(self.enc_bn2(self.enc_conv2(x))); x = self.enc_pool2(x)
        x = self.bottleneck(x)
        x = torch.relu(self.dec_up1(x)); x = torch.relu(self.dec_bn1(self.dec_conv1(x)))
        x = torch.relu(self.dec_up2(x)); x = torch.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.gap(x).view(x.size(0), -1)
        embedding = self.fc1(x) 
        return self.fc_final(self.dropout(embedding))

class BetterStudent(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(256, 128)
        self.bn_emb = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        emb = self.bn_emb(self.embedding(x))
        if return_embeddings: return emb
        return self.classifier(emb)


# Initialize & Load Teachers
T1_CLASSES = 5591  #Hard coded based on dataset split for LFW
T2_CLASSES = 5749 

teacher1 = TeacherNet(num_classes=T1_CLASSES).to(device) 
teacher2 = TeacherNet(num_classes=T2_CLASSES).to(device)

#reloads the model from the path and loads the weights
teacher1.load_state_dict(torch.load(teacher1_path, map_location=device, weights_only=False))
teacher2.load_state_dict(torch.load(teacher2_path, map_location=device, weights_only=False))
print("Teachers loaded successfully.")

for t in [teacher1, teacher2]:
    t.eval()
    for p in t.parameters(): p.requires_grad = False

student = BetterStudent(num_classes=num_classes).to(device)


# Knowledge Distillation Setup
ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
kl_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# Training Loop

max_epochs = 150
best_val_acc = 0 # saving criterion for best model, put higher to save time

for epoch in range(max_epochs):
    student.train()
    total_loss = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        student_logits = student(x)
        target_teacher_logits = torch.zeros_like(student_logits)
        
        with torch.no_grad():
            t1_out = teacher1(x)
            t2_out = teacher2(x)

            mask1 = (y < SPLIT_INDEX)
            mask2 = (y >= SPLIT_INDEX)

            # Assign expertise, selective part of the distilation, student focuses on the expert teacher for each sample
            if mask1.any():
                target_teacher_logits[mask1, :T1_CLASSES] = t1_out[mask1]
            if mask2.any():
                target_teacher_logits[mask2] = t2_out[mask2]

        loss_ce = ce_loss(student_logits, y)
        loss_kd = kl_loss(
            nn.functional.log_softmax(student_logits / T, dim=1),
            nn.functional.softmax(target_teacher_logits / T, dim=1)
        ) * (T * T)

        loss = (1 - alpha) * loss_ce + alpha * loss_kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation
    student.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            p = student(x).argmax(dim=1)
            preds.extend(p.cpu().numpy())
            trues.extend(y.cpu().numpy())

    val_acc = accuracy_score(trues, preds)
    print(f"Epoch {epoch+1}/{max_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(student.state_dict(), student_path)
        print(f"  --> Saved Best Student with Validation Accuracy: {best_val_acc:.4f}")

print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")
