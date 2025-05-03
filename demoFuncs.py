# This file is used to load models, make predictions, and return the annotated image for display the MyGUI.py file.
# It uses YOLOv8 to detect objects in the image, and then my model loads the pretrained waits saved earlier and makes predictions.

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from models.evenbetternet import EvenBetterNet
import torchvision.transforms as transforms
from ultralytics import YOLO

# This function loads my pre-trained model and the YOLOv8 model
def loadModel():
    model1 = EvenBetterNet()
    model1.load_state_dict(torch.load('best_modelReLU.pt', map_location=torch.device('cpu')))
    model1.eval()
    model2 = YOLO('YOLOv8n.pt')

    return model1, model2

# This function preprocesses the image to be compatible with the model input, and normallizes the image for prediction
def preprocessImage(image):
    # Resize the image to 224x224 and normalize it
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)
    return image

# This function takes an image path as input, loads the models, makes predictions, and draws bounding boxes around the detected objects
def predictImage(imgPath):
    # List of NBA teams to map the predicted labels
    teams = ["76ers", "Bucks", "Bulls", "Cavaliers", "Celtics", "Clippers", "Grizzlies", "Hawks", "Heat", "Hornets", "Jazz", "Kings", "Knicks", "Lakers", "Magic", "Mavericks", "Nets", "Nuggets", "Pacers", "Pelicans", "Pistons", "Raptors", "Rockets", "Spurs", "Suns", "Thunder", "Timberwolves", "Trail Blazers", "Warriors", "Wizards"]
    # Load the models, make the predictions, and draw the bounding boxes
    model1, model2 = loadModel()
    results = model2(imgPath)[0]
    img = Image.open(imgPath).convert('RGB')
    boxes = results.boxes.xyxy.cpu().numpy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=64)

    # Creating a list to store predictions
    predictions = []

    # Looping through the detected boxes, making the prediction, drawing the boxes and the labels on the image
    for box in boxes:
        # Extracting the coordinates of the bounding box
        x1, y1, x2, y2 = map(int, box)
        croppedImg = img.crop((x1, y1, x2, y2))
        croppedTensor = preprocessImage(croppedImg)

        # Making the prediction using my trained model
        with torch.no_grad():
            outputs = model1(croppedTensor)
        _, predicted = torch.max(outputs, 1)
        team = teams[predicted.item()]

        # Appending the prediction and the boxes coordinates to the predictions list
        predictions.append((team, (x1, y1, x2, y2)))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=6)

        # Drawing the label and the box on the image
        bbox = draw.textbbox((0, 0), team, font=font)
        width = bbox[2] - bbox[0]

        # Finding the middle of the box to put the text at the top middle
        centerX = (x1 + x2) // 2
        textX = centerX - (width // 2)
        draw.text((textX, y1), team, fill="red", font=font)

    annotatedImg = "annotated_image.jpg"
    img.save(annotatedImg)

    return annotatedImg