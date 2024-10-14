import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Definir el modelo (CNN básica)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Transformaciones de las imágenes
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargar datos de entrenamiento (asegúrate de tener las imágenes organizadas)
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Definir el modelo, la función de pérdida y el optimizador
model = CNN()
criterion = nn.BCELoss()  # Pérdida binaria ya que estamos clasificando entre gato/no gato
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)  # Ajustar dimensiones de las etiquetas

        # Resetear gradientes
        optimizer.zero_grad()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward y optimización
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}, Pérdida: {running_loss/len(train_loader)}")

# Guardar los pesos del modelo entrenado
torch.save(model.state_dict(), 'modelo_gato.pth')
# ... código de entrenamiento ...

# Entrenar el modelo
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)  # Ajustar dimensiones de las etiquetas

        # Resetear gradientes
        optimizer.zero_grad()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward y optimización
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}, Pérdida: {running_loss/len(train_loader)}")

# **Guardar los pesos del modelo después del entrenamiento**
torch.save(model.state_dict(), 'modelo_gato.pth')
