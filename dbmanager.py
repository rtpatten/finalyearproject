import psycopg2
import urllib.parse as urlparse
import os
import urllib3
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import json
from datetime import date, timedelta
 
def main():
    #Execute function to predict todays games
    my_token = 'jjgBB7667hbs'
    http = urllib3.PoolManager()
 
    # Load features used by lgb to be called at prediction
    cols_to_use_winner = np.load('saved_files/cols_to_use_winner.npy')
    cols_to_use_uo = np.load('saved_files/cols_to_use_uo.npy')
    cols_to_use_gng = np.load('saved_files/cols_to_use_gng.npy')
 
    # Define today's date or select another date for manual testing
    date_today = pd.Timestamp(date.today()+ timedelta(days=1))
    #date_today = pd.Timestamp('today').normalize()
    #date_today = pd.Timestamp(2020, 2, 22, 0)
 
    # Initiate dataframe to save in database
    cols_to_store = ['match_id', 'league_id', 'season_id', 'season_name',
                     'date', 'time', 'localteam_id', 'visitorteam_id',
                     'home_win_prob', 'draw_prob', 'away_win_prob',
                     'over_prob', 'under_prob', 'goal_prob', 'no_goal_prob']
 
    predictions_df = pd.DataFrame(columns=cols_to_store)
 
    for temp_season in [16222]:
 
        # Select season for which to bring data
        request_data = http.request('GET', 'https://soccer.sportmonks.com/api/v2.0/seasons/' + str(
            temp_season) + '?api_token=' + my_token + '&amp;amp;amp;amp;amp;amp;amp;amp;amp;include=fixtures')
        data_dict = json.loads(request_data.data.decode('UTF-8'))
 
        # Initiate dataframe
        df = pd.DataFrame(index=range(len(data_dict['data']['fixtures']['data'])),
                          columns=['match_id', 'league_id', 'season_id', 'season_name', 'round_id', 'stage_id',
                                   'date', 'time', 'venue_id', 'referee_id',
                                   'localteam_id', 'visitorteam_id', 'home_standing', 'away_standing',
                                   'home_goals', 'away_goals', 'home_coach', 'away_coach',
                                   'weather_type', 'weather_temp', 'weather_clouds',
                                   'weather_humidity', 'weather_windspeed'
                                   ])
 
        for i in range(len(data_dict['data']['fixtures']['data'])):
            try:
                df['season_name'][i] = data_dict['data']['name']
                df['league_id'][i] = data_dict['data']['fixtures']['data'][i]['league_id']
                df['match_id'][i] = data_dict['data']['fixtures']['data'][i]['id']
                df['season_id'][i] = data_dict['data']['fixtures']['data'][i]['season_id']
                df['round_id'][i] = data_dict['data']['fixtures']['data'][i]['round_id']
                df['stage_id'][i] = data_dict['data']['fixtures']['data'][i]['stage_id']
                df['date'][i] = data_dict['data']['fixtures']['data'][i]['time']['starting_at']['date']
                df['time'][i] = data_dict['data']['fixtures']['data'][i]['time']['starting_at']['time']
                df['venue_id'][i] = data_dict['data']['fixtures']['data'][i]['venue_id']
                df['referee_id'][i] = data_dict['data']['fixtures']['data'][i]['referee_id']
                df['localteam_id'][i] = data_dict['data']['fixtures']['data'][i]['localteam_id']
                df['visitorteam_id'][i] = data_dict['data']['fixtures']['data'][i]['visitorteam_id']
                df['home_standing'][i] = data_dict['data']['fixtures']['data'][i]['standings']['localteam_position']
                df['away_standing'][i] = data_dict['data']['fixtures']['data'][i]['standings']['visitorteam_position']
                df['home_goals'][i] = data_dict['data']['fixtures']['data'][i]['scores']['localteam_score']
                df['away_goals'][i] = data_dict['data']['fixtures']['data'][i]['scores']['visitorteam_score']
                df['home_coach'][i] = data_dict['data']['fixtures']['data'][i]['coaches']['localteam_coach_id']
                df['away_coach'][i] = data_dict['data']['fixtures']['data'][i]['coaches']['visitorteam_coach_id']
                df['weather_type'][i] = data_dict['data']['fixtures']['data'][i]['weather_report']['type']
                df['weather_temp'][i] = data_dict['data']['fixtures']['data'][i]['weather_report']['temperature'][
                    'temp']
                df['weather_clouds'][i] = data_dict['data']['fixtures']['data'][i]['weather_report']['clouds']
                df['weather_humidity'][i] = data_dict['data']['fixtures']['data'][i]['weather_report']['humidity']
                df['weather_windspeed'][i] = data_dict['data']['fixtures']['data'][i]['weather_report']['wind']['speed']
            except:
                pass
 
        # Save time as 15:00 instead of 15:00:00
        df['time'] = df['time'].str[:5]
 
        # Sort dataframe by round_id
        df['round_id'] = df['round_id'].astype('int')
        df = df.sort_values(by='round_id').reset_index(drop=True)
 
        # Create matchday from round_id
        df['matchday'] = 1
        round1 = df['round_id'][0]
        mday = 1
        for i in range(1, df.shape[0]):
            if df['round_id'][i] &amp;amp;amp;amp;amp;amp;amp;amp;gt; round1:
                df['matchday'][i] = mday + 1
                mday += 1
                round1 = df['round_id'][i]
            else:
                df['matchday'][i] = mday
 
        # Find game winner
        df['winner'] = 'home'
        for i in range(df.shape[0]):
            if df['home_goals'][i] == df['away_goals'][i]:
                df['winner'][i] = 'draw'
            elif df['home_goals'][i] &amp;amp;amp;amp;amp;amp;amp;amp;lt; df['away_goals'][i]:
                df['winner'][i] = 'away'
 
        # Calculate under-over
        df['under_over'] = df.apply(lambda x: 'under' if (x['home_goals'] + x['away_goals']) &amp;amp;amp;amp;amp;amp;amp;amp;lt; 3 else 'over', axis=1)
 
        # ----------------------Calculate all goals in favor of each team------------------
        all_goals_for = pd.DataFrame(np.zeros((df['matchday'].max() + 1, df['localteam_id'].nunique())))
        all_goals_for.columns = df['localteam_id'].unique()
 
        for t, team in enumerate(df['localteam_id'].unique()):
 
            standings_team = df[(df['localteam_id'] == team) | (df['visitorteam_id'] == team)].reset_index(drop=True)
 
            # Calculate goals in favor and goals against home team
            standings_team['goals_of_game'] = 0
 
            for i in range(standings_team.shape[0]):
                if (standings_team['localteam_id'].iloc[i] == team):
                    standings_team['goals_of_game'].iloc[i] = standings_team['home_goals'].iloc[i]
                else:
                    standings_team['goals_of_game'].iloc[i] = standings_team['away_goals'].iloc[i]
 
                standings_team['goals_sum'] = standings_team['goals_of_game'].cumsum()
 
                # If len(standings_team['goals_sum'].values)&amp;amp;amp;amp;amp;amp;amp;amp;lt;len(all_goals_for.iloc[1:,t]) then we
                # will forward fill the short array to make them equal in length
                f = list(standings_team['goals_sum'].values)
                g = all_goals_for.iloc[1:, t].values
                if len(f) &amp;amp;amp;amp;amp;amp;amp;amp;lt; len(g):
                    len_diff = len(g) - len(f)
                    last_f_value = f[-1]
                    fillin_values = [last_f_value] * len_diff
                    f.extend(fillin_values)
                elif len(g) &amp;amp;amp;amp;amp;amp;amp;amp;lt; len(f):
                    f = f[-len(g):]
 
                all_goals_for.iloc[1:, t] = f
 
        # Add home goals to initial dataframe
        df['home_goals_for_before_game'] = 0
        for i in range(df.shape[0]):
            df['home_goals_for_before_game'][i] = all_goals_for.loc[df['matchday'][i] - 1, df['localteam_id'][i]]
 
        # Add away goals to initial dataframe
        df['away_goals_for_before_game'] = 0
        for i in range(df.shape[0]):
            df['away_goals_for_before_game'][i] = all_goals_for.loc[df['matchday'][i] - 1, df['visitorteam_id'][i]]
 
        # ----------------------Calculate all goals against each team------------------
        all_goals_against = pd.DataFrame(np.zeros((df['matchday'].max() + 1, df['localteam_id'].nunique())))
        all_goals_against.columns = df['localteam_id'].unique()
 
        for t, team in enumerate(df['localteam_id'].unique()):
 
            standings_team = df[(df['localteam_id'] == team) | (df['visitorteam_id'] == team)].reset_index(drop=True)
 
            # Calculate goals in favor and goals against home team
            standings_team['goals_of_game'] = 0
 
            for i in range(standings_team.shape[0]):
                if (standings_team['localteam_id'].iloc[i] == team):
                    standings_team['goals_of_game'].iloc[i] = standings_team['away_goals'].iloc[i]
                else:
                    standings_team['goals_of_game'].iloc[i] = standings_team['home_goals'].iloc[i]
 
                standings_team['goals_sum'] = standings_team['goals_of_game'].cumsum()
 
                # If len(standings_team['goals_sum'].values)&amp;amp;amp;amp;amp;amp;amp;amp;lt;len(all_goals_for.iloc[1:,t]) then we
                # will forward fill the short array to make them equal in length
                f = list(standings_team['goals_sum'].values)
                g = all_goals_against.iloc[1:, t].values
                if len(f) &amp;amp;amp;amp;amp;amp;amp;amp;lt; len(g):
                    len_diff = len(g) - len(f)
                    last_f_value = f[-1]
                    fillin_values = [last_f_value] * len_diff
                    f.extend(fillin_values)
                elif len(g) &amp;amp;amp;amp;amp;amp;amp;amp;lt; len(f):
                    f = f[-len(g):]
 
                all_goals_against.iloc[1:, t] = f
 
        # Add home goals (against) to initial dataframe
        df['home_goals_against_before_game'] = 0
        for i in range(df.shape[0]):
            df['home_goals_against_before_game'][i] = all_goals_against.loc[
                df['matchday'][i] - 1, df['localteam_id'][i]]
 
        # Add away goals to initial dataframe
        df['away_goals_against_before_game'] = 0
        for i in range(df.shape[0]):
            df['away_goals_against_before_game'][i] = all_goals_against.loc[
                df['matchday'][i] - 1, df['visitorteam_id'][i]]
 
        # ----------------Calculate points of both teams before game----------------
        df['home_points'] = 0
        df['away_points'] = 0
 
        for i in range(df.shape[0]):
            if df['home_goals'].iloc[i] &amp;amp;amp;amp;amp;amp;amp;amp;gt; df['away_goals'].iloc[i]:
                df['home_points'].iloc[i] = 3
                df['away_points'].iloc[i] = 0
            elif df['home_goals'].iloc[i] &amp;amp;amp;amp;amp;amp;amp;amp;lt; df['away_goals'].iloc[i]:
                df['home_points'].iloc[i] = 0
                df['away_points'].iloc[i] = 3
            elif df['home_goals'].iloc[i] == df['away_goals'].iloc[i]:
                df['home_points'].iloc[i] = 1
                df['away_points'].iloc[i] = 1
 
        all_points = pd.DataFrame(np.zeros((df['matchday'].max() + 1, df['localteam_id'].nunique())))
        all_points.columns = df['localteam_id'].unique()
 
        for t, team in enumerate(df['localteam_id'].unique()):
 
            standings_team = df[(df['localteam_id'] == team) | (df['visitorteam_id'] == team)].reset_index(drop=True)
 
            # Calculate goals in favor and goals against home team
            standings_team['points'] = 0
 
            for i in range(standings_team.shape[0]):
                if (standings_team['localteam_id'].iloc[i] == team):
                    standings_team['points'].iloc[i] = standings_team['home_points'].iloc[i]
                else:
                    standings_team['points'].iloc[i] = standings_team['away_points'].iloc[i]
 
                standings_team['points_sum'] = standings_team['points'].cumsum()
 
                # If len(standings_team['points_sum'].values)&amp;amp;amp;amp;amp;amp;amp;amp;lt;len(all_points.iloc[1:,t]) then we
                # will forward fill the short array to make them equal in length
                f = list(standings_team['points_sum'].values)
                g = all_points.iloc[1:, t].values
                if len(f) &amp;amp;amp;amp;amp;amp;amp;amp;lt; len(g):
                    len_diff = len(g) - len(f)
                    last_f_value = f[-1]
                    fillin_values = [last_f_value] * len_diff
                    f.extend(fillin_values)
                elif len(g) &amp;amp;amp;amp;amp;amp;amp;amp;lt; len(f):
                    f = f[-len(g):]
 
                all_points.iloc[1:, t] = f
 
        # Add home points to initial dataframe
        df['home_points_before_game'] = 0
        for i in range(df.shape[0]):
            df['home_points_before_game'][i] = all_points.loc[df['matchday'][i] - 1, df['localteam_id'][i]]
 
        # Add away points to initial dataframe
        df['away_points_before_game'] = 0
        for i in range(df.shape[0]):
            df['away_points_before_game'][i] = all_points.loc[df['matchday'][i] - 1, df['visitorteam_id'][i]]
 
        # Keep only today's games
        df['date'] = pd.to_datetime(df['date'])
        df_to_pred = df[df['date'] == date_today]
 
        ###########-----------Here begins the prediction part---------#######
        if df_to_pred.shape[0] &amp;amp;amp;amp;amp;amp;amp;amp;gt; 0:
            df_to_pred = df_to_pred.copy()
 
            # Add current month
            currentMonth = datetime.now().month
            df_to_pred['month'] = currentMonth
 
            # Get time of game
            df_to_pred['hour'] = df_to_pred['time'].str[:2].astype('int16')
 
            # Impute venue_id
            df_to_pred['venue_id'] = df_to_pred['venue_id'].fillna(-1)
 
            float_cols = ['home_standing', 'away_standing', 'home_points_before_game', 'away_points_before_game']
            for col in float_cols:
                df_to_pred[col] = df_to_pred[col].astype('float16')
 
            int_cols = ['venue_id', 'localteam_id', 'visitorteam_id']
            for col in int_cols:
                df_to_pred[col] = df_to_pred[col].astype('int16')
 
            df_to_pred['standing_diff'] = df_to_pred['home_standing'] - df_to_pred['away_standing']
            df_to_pred['points_diff'] = df_to_pred['home_points_before_game'] - df_to_pred['away_points_before_game']
            df_to_pred['goal_balance_home'] = df_to_pred['home_goals_for_before_game'] - df_to_pred[
                'home_goals_against_before_game']
            df_to_pred['goal_balance_away'] = df_to_pred['away_goals_for_before_game'] - df_to_pred[
                'away_goals_against_before_game']
            df_to_pred['goal_balance_diff'] = df_to_pred['goal_balance_home'] - df_to_pred['goal_balance_away']
            df_to_pred['total_goals'] = df_to_pred['home_goals_for_before_game'] + df_to_pred[
                'away_goals_for_before_game']
            df_to_pred['goals_for_diff'] = df_to_pred['home_goals_for_before_game'] - df_to_pred[
                'away_goals_for_before_game']
            df_to_pred['goals_against_diff'] = df_to_pred['home_goals_against_before_game'] - df_to_pred[
                'away_goals_against_before_game']
 
           
 
            # Keep only useful features
            df_winner = df_to_pred[cols_to_use_winner]
            df_uo = df_to_pred[cols_to_use_uo]
            df_gng = df_to_pred[cols_to_use_gng]
 
            # Predict and do some post-processing
            league_id_temp = df_to_pred['league_id'].iloc[0]
 
            #Load models
            gbm_model_winner = lgb.Booster(model_file='models/gbm_model_game_result_' + str(league_id_temp) + '.txt')
            gbm_model_uo = lgb.Booster(model_file='models/gbm_model_under_over_' + str(league_id_temp) + '.txt')
            gbm_model_gng = lgb.Booster(model_file='models/gbm_model_gng_' + str(league_id_temp) + '.txt')
 
            # Predict and do some post-processing to turn numbers to strings
            # 1. Match winner prediction
            winner_predictions = pd.DataFrame(gbm_model_winner.predict(df_winner),
                                              columns=['home_win_prob', 'draw_prob', 'away_win_prob'])
            # 2. Under-Over prediction
            under_over_predictions = pd.DataFrame(gbm_model_uo.predict(df_uo), columns=['over_prob'])
            under_over_predictions['under_prob'] = 1 - under_over_predictions['over_prob']
            # 3. Goal-NoGoal prediction
            gng_predictions = pd.DataFrame(gbm_model_gng.predict(df_gng), columns=['goal_prob'])
            gng_predictions['no_goal_prob'] = 1 - gng_predictions['goal_prob']
 
            # Turn probability to 0-100 integer range
            winner_predictions = (winner_predictions * 100).round().astype(int)
            under_over_predictions = (under_over_predictions * 100).round().astype(int)
            gng_predictions = (gng_predictions * 100).round().astype(int)
 
            # Add predictions to df
            df_to_pred = df_to_pred.reset_index(drop=True)
            df_to_pred = df_to_pred.merge(winner_predictions, left_index=True, right_index=True)
            df_to_pred = df_to_pred.merge(under_over_predictions, left_index=True, right_index=True)
            df_to_pred = df_to_pred.merge(gng_predictions, left_index=True, right_index=True)
 
            predictions_df = pd.concat([predictions_df, df_to_pred[cols_to_store]], axis=0)
            predictions_df.reset_index(drop=True, inplace=True)
 
    if predictions_df.shape[0]&gt;0:
        # Create two new columns with the names of the teams
        team_dict = np.load('saved_files/team_dictionary.npy', allow_pickle=True).item()
        predictions_df['localteam_name'] = predictions_df['localteam_id'].map(team_dict)
        predictions_df['visitorteam_name'] = predictions_df['visitorteam_id'].map(team_dict)
 
        # Create a new column with the name of the League
        league_dict = np.load('saved_files/league_dictionary.npy', allow_pickle=True).item()
        predictions_df['league_name'] = predictions_df['league_id'].map(league_dict)
 
        # Create a new column with the iso name of the country (to show flags)
        league_country_dict = np.load('saved_files/league_country_dictionary.npy', allow_pickle=True).item()
        predictions_df['country_iso2'] = predictions_df['league_id'].map(league_country_dict)
 
        #Connect to database and write
        url = urlparse.urlparse(os.environ['dbmodel'])
        dbname = url.path[1:]
        user = url.username
        password = url.password
        host = url.hostname
        port = url.port
 
        con = psycopg2.connect(
                    dbname=dbname,
                    user=user,
                    password=password,
                    host=host,
                    port=port
                    )
 
        cur = con.cursor()
 
        for i in range(predictions_df.shape[0]):
            try:
                sql_command = '''
                INSERT INTO todays_predictions_all_leagues_v2 (match_id, league_id, season_id, season_name,
                                                            date, time, localteam_id, visitorteam_id,
                                                            home_win_prob, draw_prob, away_win_prob,
                                                            over_prob, under_prob, goal_prob, no_goal_prob,
                                                            localteam_name,visitorteam_name,league_name,
                                                            country_iso2)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s);'''
 
                sql_data = (predictions_df.iloc[i][0],predictions_df.iloc[i][1],
                            predictions_df.iloc[i][2], predictions_df.iloc[i][3],
                            predictions_df.iloc[i][4], predictions_df.iloc[i][5],
                            predictions_df.iloc[i][6], predictions_df.iloc[i][7],
                            predictions_df.iloc[i][8], predictions_df.iloc[i][9],
                            predictions_df.iloc[i][10], predictions_df.iloc[i][11],
                            predictions_df.iloc[i][12], predictions_df.iloc[i][13],
                            predictions_df.iloc[i][14], predictions_df.iloc[i][15],
                            predictions_df.iloc[i][16], predictions_df.iloc[i][17],
                            predictions_df.iloc[i][18]
                            )
                cur.execute(sql_command, sql_data)
                con.commit()
            except:
                pass
 
        #cur.close()
 
if __name__ == '__main__':
   main()