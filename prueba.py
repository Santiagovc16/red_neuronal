import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Definir el modelo (esto debe coincidir con la arquitectura del modelo entrenado)
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

# Función para cargar y preprocesar la imagen
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
def predict_image(model, image_path):
    model.eval()  # Modo de evaluación
    image = load_image(image_path)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.round(output).item()  # 0.0 o 1.0

    if prediction == 1.0:
        return "Es un gato 🐱"
    else:
        return "No es un gato 🚫"

# Cargar el modelo entrenado
model = CNN()
model.load_state_dict(torch.load('modelo_gato.pth'))  # Asegúrate de tener el archivo .pth con los pesos
model.eval()

# Pide la ruta de la imagen al usuario
image_path = input("Por favor, sube la ruta de la imagen que deseas clasificar: ")

# Realiza la predicción
resultado = predict_image(model, image_path)
print(resultado)
