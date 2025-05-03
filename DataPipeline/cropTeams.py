# This script crops every person in the images scraped in the scrapeTeams.py
# file using the YOLOv8 object detector

import os
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')

baseDir = "D:/SeniorProject/TempImages"

for team in os.listdir(baseDir):
  teamDir = os.path.join(baseDir, team)
  for image in os.listdir(teamDir):
    imageDir = os.path.join(teamDir, image)
    try:
      img = Image.open(imageDir)
    except:
      print(f"Problem with file: {imageDir} - removing this file from dataset")
      os.remove(imageDir)

def cropEveryPerson(inputDir, outputDir):
  # Make the new output directory if it doesn't exist
  if not os.path.exists(outputDir):
    os.mkdir(outputDir)

  # Looping through every team directory to crop the images
  for root, _, files in os.walk(inputDir):

    currentPath = os.path.relpath(root, inputDir)
    outputPath = os.path.join(outputDir, currentPath)

    # Making team directory in new output directory if it doesn't exist
    if not os.path.exists(outputPath):
      os.makedirs(outputPath)

    # Implementing an object detector on every image
    for file in files:

      imgPath = os.path.join(root, file)
      img = cv2.imread(imgPath)

      results = model(img)

      # Since I am passing one image at a time, I need to access the first result
      # and get the boxes in an xyxy format. I am convering the boxes to a numpy array
      # since the tensors can become computationally expensive.
      boxes = results[0].boxes.xyxy.cpu().numpy()

      # If more than one person is detected, then crop all of them to individual images
      for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        croppedImg = img[int(y1):int(y2), int(x1):int(x2)]
        cv2.imwrite(os.path.join(outputPath, f"{file}_{i}.jpg"), croppedImg)

inputBase = "D:/SeniorProject/TempImages"
outputBase = "D:/SeniorProject/CroppedTempImages"

cropEveryPerson(inputBase, outputBase)