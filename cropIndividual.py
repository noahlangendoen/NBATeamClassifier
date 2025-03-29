# This script crops the images from the individual player images scraped in 
# the scrapeIndividual.py file using the YOLOv8 object detector.

import os
from PIL import Image
from ultralytics import YOLO
import cv2

# Loading the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the images to ensure there will be no problems with object detection
baseDir = "D:/SeniorProject/UseableImages"

for team in os.listdir(baseDir):
  teamDir = os.path.join(baseDir, team)
  for image in os.listdir(teamDir):
    imageDir = os.path.join(teamDir, image)
    try:
      img = Image.open(imageDir)
    except:
      print(f"Problem with file: {imageDir} - removing this file from dataset")
      os.remove(imageDir)


def cropImages(inputDir, outputDir):
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

      boxes = results[0].boxes.xyxy.cpu().numpy()
      # If one person is detected, then crop to that person
      if len(boxes) == 1:
        x1, y1, x2, y2 = boxes[0]
        croppedImg = img[int(y1):int(y2), int(x1):int(x2)]
        cv2.imwrite(os.path.join(outputPath, file), croppedImg)
      # Otherwise, crop an image for the largest bounding box
      elif len(boxes) > 1:
        largeBox = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        x1, y1, x2, y2 = map(int, largeBox)
        croppedImg = img[int(y1):int(y2), int(x1):int(x2)]
        cv2.imwrite(os.path.join(outputPath, file), croppedImg)


inputBase = "D:/SeniorProject/UseableImages"
outputBase = "D:/SeniorProject/CroppedImages"

# Executing the function above on all the images in the UseableImages directory
cropImages(inputBase, outputBase)