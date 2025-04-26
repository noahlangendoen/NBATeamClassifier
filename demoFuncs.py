from PIL import Image, ImageDraw
import numpy as np
import torch
from models.evenbetternet import EvenBetterNet
import torchvision.transforms as transforms
from ultralytics import YOLO

def loadModel():
    model1 = EvenBetterNet()
    model1.load_state_dict(torch.load('best_modelReLU.pt', map_location=torch.device('cpu')))
    model1.eval()
    model2 = YOLO('YOLOv8n.pt')

    return model1, model2

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
    model1, model2 = loadModel()
    image = preprocessImage(imgPath)
    with torch.no_grad():
        outputs = model1(image)

    _, predicted = torch.max(outputs, 1)
    team = teams[predicted.item()]
    
    results = model2(imgPath)[0]
    img = Image.open(imgPath).convert('RGB')
    boxes = results.boxes.xyxy.cpu().numpy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    annotatedImg = "annotated_image.jpg"
    img.save(annotatedImg)  # Save the annotated image to disk

    return team, annotatedImg