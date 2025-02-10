import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image

# Setting up the base directory for image resizing
baseDir = "UseableImages/"
os.remove("UseableImages/.DS_Store")
# Looping through every team and image in the team's directory and resizing them
for team in os.listdir(baseDir):
  teamDir = os.path.join(baseDir, team)
  for image in os.listdir(teamDir):
    imageDir = os.path.join(teamDir, image)
    try: 
      img = Image.open(imageDir)
      resizedImage = img.resize((224,224))
      resizedImage.save(imageDir)
    except:
      print(f"Problem with file: {imageDir} - removing this file from dataset")
      os.remove(imageDir)