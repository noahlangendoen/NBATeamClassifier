# This script scrapes images of current NBA players from Bing Images
# and stores them in their respective teams directory

import requests
import pandas as pd
import os
from bing_image_downloader import downloader

# Link for gathering all NBA players statistics all time
player_index = "https://stats.nba.com/stats/playerindex?College=&Country=&DraftPick=&DraftRound=&DraftYear=&Height=&Historical=1&LeagueID=00&Season=2024-25&SeasonType=Regular%20Season&TeamID=0&Weight="

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Origin": "https://stats.nba.com",
    "Referer": "https://stats.nba.com/stats/playerindex"
}

response = requests.get(player_index, headers=headers)

# Checking if scrape attempt worked successfully and storing the data
if response.status_code == 200:

    data = response.json()
    result_sets = data['resultSets'][0]
    headers = result_sets['headers']
    row_set = result_sets['rowSet']

else:

    print("Data scrapping failed.")

# Storing the scraped data in a pandas dataframe and creating a secondary dataframe of only current NBA players
df = pd.DataFrame(row_set, columns=headers)
df_current = df[df['ROSTER_STATUS'] == 1.0]
df_current.reset_index(drop=True, inplace=True)
df_current = df_current.drop(columns=['IS_DEFUNCT', 'STATS_TIMEFRAME', 'ROSTER_STATUS', 'PLAYER_SLUG', 'TEAM_SLUG'])
df_current['DRAFT_NUMBER'] = df_current['DRAFT_NUMBER'].fillna('Undrafted')
df_current['DRAFT_ROUND'] = df_current['DRAFT_ROUND'].fillna('Undrafted')

# Creating a dict for labeling teams for future training
team_labels = {}
count = 0
for team in df_current['TEAM_NAME'].unique():
    team_labels[team] = count
    count += 1

# Creating directory to store images for future scraping
os.makedirs("D:/SeniorProject/BingImages", exist_ok=True)

# Creating a unique directory for each team
for team in df_current['TEAM_NAME'].unique():
    os.makedirs(f"D:/SeniorProject/BingImages/{team}", exist_ok=True)

# Scraping 10 images for each player listed in the dataframe and storing them in their respective teams directory
for idx, row in df_current.iterrows():
    search = f"{row['PLAYER_FIRST_NAME']} {row['PLAYER_LAST_NAME']} {row['TEAM_NAME']}"
    file_path = f"D:/SeniorProject/BingImages/{row['TEAM_NAME']}/"
    print(f"Currently on player {idx + 1} of {len(df_current)}")
    downloader.download(search, limit=10, output_dir=file_path, adult_filter_off=True, force_replace=False, timeout=60)

# Removing files that do not contain the proper extension for training
valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
for team in df_current['TEAM_NAME'].unique():
  folderpath = f"D:/SeniorProject/BingImages/{team}/"
  for directory in os.listdir(folderpath):
    for filename in os.listdir(folderpath + directory):
      name, extension = os.path.splitext(filename)
      if extension not in valid_extensions:
        os.remove(folderpath + directory + "/" + filename)
        print(f'Removed {filename}')