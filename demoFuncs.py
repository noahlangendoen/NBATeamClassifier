from PIL import Image
import numpy as np
import torch
from models.evenbetternet import EvenBetterNet
import torchvision.transforms as transforms

def loadModel():
    model = EvenBetterNet()
    model.load_state_dict(torch.load('EvenBetterNet.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocessImage(imgPath):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(imgPath).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predictImage(imgPath):
    teams = ["76ers", "Bucks", "Bulls", "Cavaliers", "Celtics", "Clippers", "Grizzlies", "Hawks", "Heat", "Hornets", "Jazz", "Kings", "Knicks", "Lakers", "Magic", "Mavericks", "Nets", "Nuggets", "Pacers", "Pelicans", "Pistons", "Raptors", "Rockets", "Spurs","Suns", "Thunder", "Timberwolves", "Trail Blazers", "Warriors", "Wizards"]
    model = loadModel()
    image = preprocessImage(imgPath)
    with torch.no_grad():
        outputs = model(image)

    _, predicted = torch.max(outputs, 1)
  
    return teams[predicted.item()]