import http.client
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Example usage:

extract = download_data('PL', '2020-08-01', '2021-08-01')
raw_df = create_df(extract)
processed_df = preprocess_df(raw_df)

'''


my_token = '9a9eaeaf20d248bf8f0cce4ee3b0445b'

# Uses the following api API: https://www.football-data.org/
def download_data(league_name, date_min, date_max):

	# Save some stuff
    league_ids = {"BSA": 2013,"BL": 2002,"FL1": 2015,"PL": 2021,"ELC": 2016,
    			  "PD": 2014,"SA": 2019,"PPL": 2017,"DED": 2003,"CL": 2001}

    # Access token for API
	my_token = '9a9eaeaf20d248bf8f0cce4ee3b0445b'


    # Check input
    if league_name not in list(league_ids):
        raise ValueError('team_name must be in: '+str(list(league_ids))[1:-1])
    if (len(date_min) != 10) or (len(date_max) != 10):
        raise ValueError('dates must be in form: 20XX-XX-XX')
    
    # Create query
    query='/v2/competitions/{}/matches?dateFrom={}&dateTo={}'.format(league_ids[league_name], date_min, date_max) 
    
    # Get data
    connection = http.client.HTTPConnection('api.football-data.org')
    headers = { 'X-Auth-Token': my_token }
    connection.request('GET', query, None, headers)
    json_data = json.loads(connection.getresponse().read().decode())
    
    return json_data
        
        

# Get columns from json for each match
def extract_from_json(single_match):
    # Get all relevant info
    match = {}
    match['match_id'] = single_match['id']
    match['stage'] = single_match['stage']
    match['team_h_id'] = single_match['homeTeam']['id']
    match['team_a_id'] = single_match['awayTeam']['id']
    match['team_h_name'] = single_match['homeTeam']['name']
    match['team_a_name'] = single_match['awayTeam']['name']  
    match['goals_h'] = single_match['score']['fullTime']['homeTeam']
    match['goals_a'] = single_match['score']['fullTime']['awayTeam']
    match['season_id'] = single_match['season']['id']
    match['matchday'] = single_match['matchday']
    match['date'] = single_match['utcDate']
    return match



# Create a row per team per game with correct info
def split_games(match):
    
    # Function for cleaning team name
    def clean_team_name(team):
        cleaned = team.replace(' ','_').replace('_FC','').replace('_&','').lower()
        return cleaned
    # Home datapoint
    score_h = {}
    score_h['team'] = clean_team_name(match['team_h_name'])
    score_h['opponent'] = clean_team_name(match['team_a_name'])
    score_h['home'] = 1
    score_h['goals_scored'] = match['goals_h']
    score_h['goals_conceded'] = match['goals_a']
    score_h['date'] = match['date']
    # Away datapoint
    score_a = {}
    score_a['team'] = clean_team_name(match['team_a_name'])
    score_a['opponent'] = clean_team_name(match['team_h_name'])
    score_a['home'] = 0
    score_a['goals_scored'] = match['goals_a']
    score_a['goals_conceded'] = match['goals_h']
    score_a['date'] = match['date']

    return score_h, score_a
 
    
# Loop through data and turn to dataframe
def create_df(response):
    df = pd.DataFrame()
    for r in response['matches']:
        samples = split_games(extract_from_json(r))
        df = df.append(samples[0], ignore_index = True)
        df = df.append(samples[1], ignore_index = True)
    df['date'] = pd.to_datetime(df['date'])
    return df





