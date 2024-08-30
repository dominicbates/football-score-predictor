import http.client
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

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
    league_ids = {"BSA": 2013,"BL1": 2002,"FL1": 2015,"PL": 2021,"ELC": 2016,
                  "PD": 2014,"SA": 2019,"PPL": 2017,"DED": 2003,"CL": 2001}

    # Check input
    if league_name not in list(league_ids):
        raise ValueError('team_name must be in: '+str(list(league_ids))[1:-1])
    if (len(date_min) != 10) or (len(date_max) != 10):
        raise ValueError('dates must be in form: 20XX-XX-XX')
    
    # Create query
    query='/v2/competitions/{}/matches?dateFrom={}&dateTo={}'.format(league_ids[league_name], date_min, date_max) 
    # SHOULD CHANGE TO V4 AT SOME POINT


    # Get response and check if it worked
    response_extracted = False
    while response_extracted==False:
        response = get_query(query)
        if 'message' in list(response):
            print('Found error message in repsonse, trying again in 30 sec')
            print(response['message'])
            time.sleep(30)
        else:
            response_extracted=True

    return response
        
            


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
    # print(response)
    df = pd.DataFrame()

    if response['count'] > 0:
        for r in response['matches']:
            samples = split_games(extract_from_json(r))
            # df = df.append(samples[0], ignore_index = True) # A bit inefficient but it's all good
            # df = df.append(samples[1], ignore_index = True)
            df = pd.concat([df, pd.DataFrame(samples[0], index=[0])], ignore_index=True) # A bit inefficient but it's all good
            df = pd.concat([df, pd.DataFrame(samples[1], index=[0])], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
    return df


# Create features from dataframe (and one-hot encode)
def preprocess_df(scores_df):
    
    # Blank dataframe
    processed_df = pd.DataFrame()
    processed_df['date'] = pd.to_datetime(scores_df['date']) # Convert from str to datetime
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
    
    return processed_df
    


def add_recency_weight(processed_df):

    # Weight for recent games (goes from 0 in oldest game to 1 in most recent game)
    processed_df = processed_df.sort_values(by=['date'], ascending=True, ignore_index=True)
    processed_df['f|recency_weight'] = 0

    m_played = processed_df['played'] == 1
    processed_df.loc[m_played,'f|recency_weight'] = np.linspace(0,1,m_played.sum())

    return processed_df


# def get_production_data(n_years_minus=2, n_weeks_plus=8, force_current_date=None):
    
#     if force_current_date is None:
#         today = datetime.date.today()
#     else:
#         today = force_current_date
        
#     # Loop through different years and store data
#     processed_dataframes = []
#     for n in range(n_years_minus):
#         min_date =  today - datetime.timedelta(weeks = (n+1)*52)
#         max_date =  today - datetime.timedelta(weeks = n*52)
#         # Get data
#         extract = download_data('PL', min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
#         # print(extract)
#         if extract['count'] > 0:
#             raw_df = create_df(extract)
#             processed_df = preprocess_df(raw_df)
#             processed_dataframes.append(processed_df)
#             time.sleep(5) # Just incase
        
#     # Add future games
#     max_date =  today + datetime.timedelta(weeks = n_weeks_plus)

#     extract = download_data('PL', today.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
#     if extract['count'] > 0:
#         raw_df = create_df(extract)
#         processed_df = preprocess_df(raw_df)
#         processed_dataframes.append(processed_df)

#     # Combine datasets
#     production_df = pd.concat(processed_dataframes, ignore_index=True).drop_duplicates(ignore_index=True)
#     # Incase different columns?
#     production_df = production_df.fillna(0)

#     # Fake the played feature if required (set to -1)
#     # if force_current_date is None:
#     #     None
#     # else:
#     m_fake_unplayed = (production_df['played'] == 1) & (production_df['date'].dt.date > today)
#     production_df.loc[m_fake_unplayed,'played'] = -1
#     production_df = add_recency_weight(production_df)

#     return production_df


def get_production_data(league = 'PL', n_weeks_minus=52, n_weeks_plus=8, force_current_date=None):
    
    if force_current_date is None:
        today = datetime.date.today()
    else:
        today = force_current_date
        
    # Get dates for query
    min_date =  today - datetime.timedelta(weeks = n_weeks_minus)
    max_date =  today + datetime.timedelta(weeks = n_weeks_plus)

    # Get data
    extract = download_data(league, min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
    # print(extract)
    if extract['count'] > 0:
        raw_df = create_df(extract)
        production_df = preprocess_df(raw_df)
        time.sleep(1) # Just incase
    else:
        print('NO RESULTS FOUND!!!')
 
     # Incase 
    production_df = production_df.fillna(0)

    # Fake the played feature if required (set to -1)
    m_fake_unplayed = (production_df['played'] == 1) & (production_df['date'].dt.date > today)
    production_df.loc[m_fake_unplayed,'played'] = -1
    production_df = add_recency_weight(production_df)

    return production_df






def apply_promoted_prior(df, mapping):
    '''
    Applies last years results to this years prompted teams 
    given a mapping passed e.g. {'leicster':'luton'}

    This acts sort of equivalently to a bayesian prior on 
    newly prompted teams, assuming their performance will 
    be similar to last years prompted teams. This prior will
    then become less informative as more data becomes 
    available as dictated by the weight curve.
    '''
    for col in ['team', 'opponent']:
        for prompted in list(mapping):
            # To edit
            m_edit = (df[col]==mapping[prompted])
            # Change team names
            df.loc[m_edit, col] = prompted
            # Change one hot encoded features (remove for old team, add for new)
            if col == 'team':
                df.loc[m_edit, 'f|team|'+prompted] = 1
                df.loc[m_edit, 'f|team|'+mapping[prompted]] = 0
            elif col == 'opponent':
                df.loc[m_edit, 'f|opp|'+prompted] = 1
                df.loc[m_edit, 'f|opp|'+mapping[prompted]] = 0
    return df
