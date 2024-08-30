import pandas as pd
from soccerapi.api import Api888Sport


def get_odds(country, league):

	# Extract odds
	api = Api888Sport()
	url = api.competitions()[country][league]
	current_odds = api.odds(url)

	# Create dict
	odds_dict = {'date':[],
	        'home_team':[],
	        'away_team':[],
	        'home_odds':[],
	        'draw_odds':[],
	        'away_odds':[]
	}

	# Turn int to odds
	def transform_odds(n):
	    return (n-1000)/1000

	# Add to dataframe
	for game in current_odds:
	    odds_dict['date'].append(game['time'])
	    odds_dict['home_team'].append(game['home_team'].lower().replace(' ','_').replace('wolves','wolverhampton'))
	    odds_dict['away_team'].append(game['away_team'].lower().replace(' ','_').replace('wolves','wolverhampton'))
	    odds_dict['home_odds'].append(transform_odds(game['full_time_result']['1']))
	    odds_dict['draw_odds'].append(transform_odds(game['full_time_result']['X']))
	    odds_dict['away_odds'].append(transform_odds(game['full_time_result']['2']))
	odds_df = pd.DataFrame(odds_dict)

	return odds_df



def append_odds_to_df(final_df, odds_df):

	# Loop through and append odds (simple join doesn't work e.g. 'west_ham' vs 'west_ham_united')
	for n in range(len(final_df)):
	    for m in range(len(odds_df)):
	        if (odds_df['home_team'][m] in final_df['home_team'][n]) and (odds_df['away_team'][m] in final_df['away_team'][n]):
	            
	            # Append odds
	            final_df.loc[n,'odds|home_win'] = odds_df.loc[m,'home_odds']
	            final_df.loc[n,'odds|draw'] = odds_df.loc[m,'draw_odds']
	            final_df.loc[n,'odds|away_win'] = odds_df.loc[m,'away_odds']
	            
	            # Calculate expected innings
	            final_df.loc[n,'p|winnings|home_win'] = final_df.loc[n,'p|home_win']* (1+final_df.loc[n,'odds|home_win'])
	            final_df.loc[n,'p|winnings|draw'] = final_df.loc[n,'p|draw']* (1+final_df.loc[n,'odds|draw'])
	            final_df.loc[n,'p|winnings|away_win'] = final_df.loc[n,'p|away_win']* (1+final_df.loc[n,'odds|away_win'])

	return final_df



