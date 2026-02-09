# mini_cnn_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ----------------------------
# 1. Daten vorbereiten
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Oxford Flowers 102 Dataset als Beispiel
dataset = datasets.Flowers102(root='./data', download=True, transform=transform)

# kleines Subset für schnelle Tests
total_size = len(dataset)
train_size = min(50, total_size - 10)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5)

# ----------------------------
# 2. Mini-CNN definieren
# ----------------------------
class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*16*16, 64)
        self.fc2 = nn.Linear(64, 5)  # nur 5 Klassen als Beispiel

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x2 = self.pool(x1)
        x3 = torch.relu(self.conv2(x2))
        x4 = self.pool(x3)
        x_flat = x4.view(-1, 32*16*16)
        x_fc = torch.relu(self.fc1(x_flat))
        out = self.fc2(x_fc)
        return out

# ----------------------------
# 3. Training Setup
# ----------------------------
device = torch.device("cpu")
model = MiniCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
train_losses = []
val_losses = []

# ----------------------------
# 4. Training Loop
# ----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # modulo 5, da wir nur 5 Klassen nutzen
        loss = criterion(outputs, labels % 5)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels % 5)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# ----------------------------
# 5. Loss Plot
# ----------------------------
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_plot.png")  # Speichert die Grafik als Datei
print("Loss plot saved as loss_plot.png")