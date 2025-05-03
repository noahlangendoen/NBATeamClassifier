# This is a script that takes in an image and submits that image
# via an API call to the LLaVa-7b model using the ollama package.
# A response is returned, and then the response is checked for a
# team prediction and the image is moved to the mapped label.

import requests
import base64
import json
import os
import ollama
from PIL import Image

baseDir = "D:/SeniorProject/CroppedTempImages"
# Looping through every team and image in the team's directory and resizing and renaming them before labelling
for team in os.listdir(baseDir):
  i = 0
  teamDir = os.path.join(baseDir, team)
  for image in os.listdir(teamDir):
    imageDir = os.path.join(teamDir, image)
    newImageDir = os.path.join(teamDir, f"{team}Temp_{i}.jpg")
    try:
      with Image.open(imageDir) as img:
        image = img.resize((224, 224))
        image.save(newImageDir)
      
      if imageDir != newImageDir:
        os.remove(imageDir)

      i += 1

    except Exception as e:
      print(f"Problem with file: {imageDir} - {e}")
      if os.path.exists(imageDir):
        os.remove(imageDir)

# Mapping possible team names to their labels returned by LLaVa
teamDict = {
    "Philadelphia 76ers": "76ers", "76ers": "76ers", "Philadelphia": "76ers",
    "Milwaukee Bucks": "Bucks", "Bucks": "Bucks", "Milwaukee": "Bucks", 
    "Chicago Bulls": "Bulls", "Bulls": "Bulls", "Chicago": "Bulls",
    "Cleveland Cavaliers": "Cavaliers", "Cavaliers": "Cavaliers", "Cleveland": "Cavaliers",
    "Boston Celtics": "Celtics", "Celtics": "Celtics", "Boston": "Celtics",
    "Los Angeles Clippers": "Clippers", "Clippers": "Clippers",
    "Memphis Grizzlies": "Grizzlies", "Grizzlies": "Grizzlies", "Memphis": "Grizzlies", 
    "Atlanta Hawks": "Hawks", "Hawks": "Hawks", "Atlanta": "Hawks",
    "Miami Heat": "Heat", "Heat": "Heat", "Miami": "Heat",
    "Charlotte Hornets": "Hornets", "Hornets": "Hornets", "Charlotte": "Hornets",
    "Utah Jazz": "Jazz", "Jazz": "Jazz", "Utah": "Jazz",
    "Sacramento Kings": "Kings", "Kings": "Kings", "Sacramento": "Kings",
    "New York Knicks": "Knicks", "Knicks": "Knicks", "New York": "Knicks",
    "Los Angeles Lakers": "Lakers", "Lakers": "Lakers", "Los Angeles": "Lakers",
    "Orlando Magic": "Magic", "Magic": "Magic", "Orlando": "Magic",
    "Dallas Mavericks": "Mavericks", "Mavericks": "Mavericks", "Dallas": "Mavericks",
    "Brooklyn Nets": "Nets", "Nets": "Nets", "Brooklyn": "Nets",
    "Denver Nuggets": "Nuggets", "Nuggets": "Nuggets", "Denver": "Nuggets", 
    "Indiana Pacers": "Pacers", "Pacers": "Pacers", "Indiana": "Pacers",
    "New Orleans Pelicans": "Pelicans", "Pelicans": "Pelicans", "New Orleans": "Pelicans",
    "Detroit Pistons": "Pistons", "Pistons": "Pistons", "Detroit": "Pistons",
    "Toronto Raptors": "Raptors", "Raptors": "Raptors", "Toronto": "Raptors",
    "Houston Rockets": "Rockets", "Rockets": "Rockets", "Houston": "Rockets",
    "San Antonio Spurs": "Spurs", "Spurs": "Spurs", "San Antonio": "Spurs",
    "Phoenix Suns": "Suns", "Suns": "Suns", "Phoenix": "Suns",
    "Oklahoma City Thunder": "Thunder", "Thunder": "Thunder", "Oklahoma City": "Thunder",
    "Minnesota Timberwolves": "Timberwolves", "Timberwolves": "Timberwolves", "Minnesota": "Timberwolves",
    "Portland Trail Blazers": "Trail Blazers", "Trail Blazers": "Trail Blazers", "Portland": "Trail Blazers",
    "Golden State Warriors": "Warriors", "Warriors": "Warriors", "Golden State": "Warriors",
    "Washington Wizards": "Wizards", "Wizards": "Wizards", "Washington": "Wizards"
}

# Function for encoding images to base64 for the LLaVa API Calls
def encodeImage(imageDir):
    # Opening the image in binary mode
    with open(imageDir, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")

# Function for the LLaVa API call and returning the response from the model
def makePrediction(imagePath):
    # Creating the ollama API URL
    ollamaAPI = "http://localhost:11434/api/generate"
    
    # Creating the payload (prompt and image path) for the LLaVa API call
    payload = {
        "model": "llava",
        "prompt": (
           "Is there an NBA Player wearing a jersey in this image?" 
            "If so, what team are they on?"
        ),
        "images": [encodeImage(imagePath)]
    }

    # Making the api call and storing the response
    response = requests.post(ollamaAPI, json=payload)
    responseText = ""

    # Reponse is returned as a JSON, so we iterate through the response lines 
    # to parse the response together until the "done" field is passed
    for chunk in response.iter_lines():
        if chunk:
            decodedChunk = json.loads(chunk.decode("utf-8"))
            responseText += decodedChunk.get("response", "")
            if decodedChunk.get("done", True):
                break

    return responseText

# List of every NBA team
teams = ["76ers", "Bucks", "Bulls", "Cavaliers", "Celtics", "Clippers",
        "Grizzlies", "Hawks", "Heat", "Hornets", "Jazz", "Kings",
        "Knicks", "Lakers", "Magic", "Mavericks", "Nets",
        "Nuggets", "Pacers", "Pelicans", "Pistons", "Raptors",
        "Rockets", "Spurs", "Suns", "Thunder", "Timberwolves",
        "Trail Blazers", "Warriors", "Wizards"]

# For every team in the NBA, moved the image to the label predicted by LLaVa-7b
for team in teams:
    teamDir = f"D:/SeniorProject/CroppedTempImages/{team}"
    for root, _, files in os.walk(teamDir):
        for file in files:
            imgPath = os.path.join(root, file)
            prediction = makePrediction(imgPath)

            mappedPrediction = teamDict.get(prediction, "unknown")
            mappedPrediction = mappedPrediction.lower()

            for key in teamDict:
                if key.lower() in prediction.lower():
                    mappedPrediction = teamDict[key]

            if mappedPrediction != "unknown":
                outputDir = f"D:/SeniorProject/LLMImageLabels/{mappedPrediction}"
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir)
                
                os.rename(imgPath, os.path.join(outputDir, file))
            else:
                if not os.path.exists("D:/SeniorProject/LLMImageLabels/Unkown"):
                    os.makedirs("D:/SeniorProject/LLMImageLabels/Unkown")

                os.rename(imgPath, "D:/SeniorProject/LLMImageLabels/Unkown/" + file)
