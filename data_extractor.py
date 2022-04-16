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


def get_query(query):

    # Access token for API
    my_token = '9a9eaeaf20d248bf8f0cce4ee3b0445b'
    # Get data
    connection = http.client.HTTPConnection('api.football-data.org')
    headers = { 'X-Auth-Token': my_token }
    connection.request('GET', query, None, headers)
    json_data = json.loads(connection.getresponse().read().decode())

    return json_data

# Uses the following api API: https://www.football-data.org/
def download_data(league_name, date_min, date_max):

    # Save some stuff
    league_ids = {"BSA": 2013,"BL": 2002,"FL1": 2015,"PL": 2021,"ELC": 2016,
                  "PD": 2014,"SA": 2019,"PPL": 2017,"DED": 2003,"CL": 2001}

    # Check input
    if league_name not in list(league_ids):
        raise ValueError('team_name must be in: '+str(list(league_ids))[1:-1])
    if (len(date_min) != 10) or (len(date_max) != 10):
        raise ValueError('dates must be in form: 20XX-XX-XX')
    
    # Create query
    query='/v2/competitions/{}/matches?dateFrom={}&dateTo={}'.format(league_ids[league_name], date_min, date_max) 
    
    return get_query(query)
        
            


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
    match['status'] = single_match['status']
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
    score_h['played'] = int(match['status'] == 'FINISHED')
    score_h['match_id'] = match['match_id']
    # Away datapoint
    score_a = {}
    score_a['team'] = clean_team_name(match['team_a_name'])
    score_a['opponent'] = clean_team_name(match['team_h_name'])
    score_a['home'] = 0
    score_a['goals_scored'] = match['goals_a']
    score_a['goals_conceded'] = match['goals_h']
    score_a['date'] = match['date']
    score_a['played'] = int(match['status'] == 'FINISHED')
    score_a['match_id'] = match['match_id']

    return score_h, score_a
 
    
# Loop through data and turn to dataframe
def create_df(response):
    df = pd.DataFrame()
    for r in response['matches']:
        samples = split_games(extract_from_json(r))
        df = df.append(samples[0], ignore_index = True) # A bit inefficient but it's all good
        df = df.append(samples[1], ignore_index = True)
    df['date'] = pd.to_datetime(df['date'])
    return df


# Create features from dataframe (and one-hot encode)
def preprocess_df(scores_df):
    
    # Blank dataframe
    processed_df = pd.DataFrame()
    processed_df['date'] = scores_df['date']
    processed_df['match_id'] = scores_df['match_id']
    processed_df['team'] = scores_df['team']
    processed_df['opponent'] = scores_df['opponent']	
    processed_df['played'] = scores_df['played']
    # List of all teams
    teams = list(set(scores_df['team']))

    # Team features
    for t in teams:
        m_team = (scores_df['team'] == t)
        processed_df['f|team|'+t] = m_team.astype(int)
    # Opponent features
    for t in teams:
        m_opp = (scores_df['opponent'] == t)
        processed_df['f|opp|'+t] = m_opp.astype(int)

    # Home/away feature
    processed_df['f|home'] = scores_df['home'].astype(int)
    # Target variable
    processed_df['p|goals|scored'] = scores_df['goals_scored']
    processed_df['p|goals|conceded'] = scores_df['goals_conceded']
    
    processed_df['f|played'] = scores_df['played']

    return processed_df
    



