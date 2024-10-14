import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets

# Definir el modelo CNN mejorado
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # Nueva capa convolucional añadida
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))  # Nueva capa convolucional
        x = x.view(-1, 256 * 9 * 9)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Transformaciones con data augmentation
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),  # Flip horizontal aleatorio
    transforms.RandomRotation(10),  # Rotaciones aleatorias
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargar datos de entrenamiento (organizados en carpetas "gatos" y "no_gatos")
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Definir el modelo, la función de pérdida y el optimizador con regularización L2
model = CNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # L2 regularization (weight decay)

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

# Guardar los pesos del modelo mejorado
torch.save(model.state_dict(), 'modelo_gato_mejorado.pth')

# Función para cargar y preprocesar la imagen para predicción
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Añadir batch size de 1
    return image

# Función para hacer la predicción
def predict_image(model, image_path, threshold=0.3):  # Cambia el umbral aquí
    model.eval()  # Modo de evaluación
    image = load_image(image_path)
    
    with torch.no_grad():
        output = model(image)
        prediction = output.item()  # Predicción sin redondeo

    if prediction > threshold:
        return "Es un gato 🐱"
    else:
        return "No es un gato 🚫"


# Cargar el modelo mejorado para predicción
model = CNN()
model.load_state_dict(torch.load('modelo_gato_mejorado.pth'))  # Cargar el modelo mejorado
model.eval()

# Pide la ruta de la imagen al usuario
image_path = input("Por favor, sube la ruta de la imagen que deseas clasificar: ")

# Realiza la predicción
resultado = predict_image(model, image_path)
print(resultado)
