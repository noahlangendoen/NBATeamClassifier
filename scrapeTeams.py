# This script scrapes images for every NBA team and stores them in their respective directories

import requests
import os
from bing_image_downloader import downloader

# List for team names and valid image extensions
teams = ["Hawks", "Celtics", "Nets", "Hornets", "Bulls", "Cavaliers", "Mavericks", "Nuggets", "Pistons", "Warriors", 
         "Rockets", "Pacers", "Clippers", "Lakers", "Grizzlies", "Heat", "Bucks", "Timberwolves", "Pelicans", "Knicks", 
         "Thunder", "Magic", "76ers", "Suns", "Trail Blazers", "Kings", "Spurs", "Raptors", "Jazz", "Wizards"]

valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG']

directory = "D:/SeniorProject/TempImages/"

if not os.path.exists(directory):
    os.mkdir(directory)

# Scraping images for each team
for team in teams:
    if not os.path.exists(directory + team):
        os.mkdir(directory + team)
    search = team + " players NBA"
    downloader.download(search, limit=100, output_dir=directory + team, adult_filter_off=True, force_replace=False, timeout=60)

# Checking to ensure only images with valid extensions are kept
for team in teams:
  folderpath = f"D:/SeniorProject/TempImages/{team}/"
  for directory in os.listdir(folderpath):
    for filename in os.listdir(folderpath + directory):
      name, extension = os.path.splitext(filename)
      if extension not in valid_extensions:
        os.remove(folderpath + directory + "/" + filename)
        print(f'Removed {filename}')

# Renaming files to a more consistent format
for team in teams:
    count = 0
    teamDir = f"D:/SeniorProject/TempImages/{team}"
    for filename in os.listdir(teamDir):
        count += 1
        os.rename(f"{teamDir}/{filename}", f"{teamDir}/{team}Team_{count}.jpg")