import MySQLdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, recall_score,precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
#Connect to Mysql database
db = MySQLdb.connect(host="192.168.0.1",
                     user="rtpatten",
                     passwd="rtpatten1",
                     db="modeldb")
 
# Select league to train
league_id = 325
df = pd.read_sql('SELECT * FROM match_history_' +str(league_id), con=db)
db.close()
 
#Select seasons
#1. Seasons that begin August and end May
seasons_to_train = [#'2009/2010','2010/2011','2011/2012','2012/2013',
                    # '2013/2014', '2014/2015',
                    # '2015/2016', '2016/2017',
                    '2017/2018','2018/2019']
#2. Seasons that begin March and end September
seasons_to_train_other_display = [
                                  # '2009','2010','2011', '2012',
                                  # '2013', '2014', '2015', '2016',
                                  '2017','2018']
seasons_total = seasons_to_train+seasons_to_train_other_display
df = df[df['season_name'].isin(seasons_total)]
 
#Time features
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['hour'] = df['time'].str[:2]
 
#Impute venue_id
df['venue_id'] = df['venue_id'].fillna(-1)
 
#Assign proper types
float_cols = ['home_standing', 'away_standing', 'home_points_before_game', 'away_points_before_game']
for col in float_cols:
    df[col] = df[col].astype('float16')
 
int_cols =['venue_id', 'month', 'hour', 'localteam_id', 'visitorteam_id', 'home_goals', 'away_goals']
for col in int_cols:
    df[col] = df[col].astype('int16')
 
#Feature engineering
df['standing_diff'] = df['home_standing']-df['away_standing']
df['points_diff'] = df['home_points_before_game'] - df['away_points_before_game']
df['goal_balance_home'] = df['home_goals_for_before_game']-df['home_goals_against_before_game']
df['goal_balance_away'] = df['away_goals_for_before_game']-df['away_goals_against_before_game']
df['goal_balance_diff'] = df['goal_balance_home'] - df['goal_balance_away']
df['total_goals'] = df['home_goals_for_before_game'] + df['away_goals_for_before_game']
df['total_goals_of_game'] = df['home_goals'] + df['away_goals']
df['goals_for_diff'] = df['home_goals_for_before_game'] - df['away_goals_for_before_game']
df['goals_against_diff'] = df['home_goals_against_before_game'] - df['away_goals_against_before_game']
df['goal_nogoal'] = df.apply(lambda x: 'Goal' if (x['home_goals']&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0 and x['away_goals']&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0) else 'No_Goal', axis = 1)
 
# #Change weather variable measures
# df['weather_clouds'] = df['weather_clouds'].str[:-1]
# df['weather_humidity'] = df['weather_humidity'].str[:-1]
# df['weather_windspeed'] = df['weather_windspeed'].str[:-4]
#
# for col in ['weather_temp','weather_clouds', 'weather_humidity', 'weather_windspeed']:
#     df[col] = df[col].astype('float16')
 
#Encode Winner, Under_Over and goal_no_goal outcomes
winner_dict = {'home':0, 'draw': 1, 'away': 2}
ou_dict = {'under':0, 'over': 1}
g_gn_dit = {'Goal':1, 'No_Goal': 0}
 
df['winner'] = df['winner'].map(winner_dict)
df['under_over'] = df['under_over'].map(ou_dict)
df['goal_nogoal'] = df['goal_nogoal'].map(g_gn_dit)
 
#----------------------Model 1: Under-Over prediction-------------------------
 
#Select features and create X and y
cols_to_use_uo = [#'venue_id',
                'month',
                'hour',
               'localteam_id', 'visitorteam_id',
               #'winner', 'under_over',
               #'home_standing', 'away_standing',
               'standing_diff', 'points_diff',
               'goal_balance_home', 'goal_balance_away',
                 #'goal_balance_diff',
               'total_goals'
               ]
 
#Save features used by lgb to be called at prediction
#np.save('cols_to_use_uo.npy', cols_to_use_uo)
 
df_final_uo = df[cols_to_use_uo]
ylabel_underover = df['under_over']
 
#Split in train and test
train_x, test_x, train_y, test_y = train_test_split(df_final_uo, ylabel_underover, shuffle=True)
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_x, train_y, shuffle=True)
 
lgb_train = lgb.Dataset(train_early_x, train_early_y)
lgb_eval = lgb.Dataset(valid_early_x, valid_early_y)
 
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    #'metric': {'auc'},
    'num_leaves': 16,
    'n_estimators': 100,
    'max_depth': -1,
    'learning_rate': 0.05,
    'bagging_fraction': 0.5,
    #'reg_alpha': 0.8,
    #'reg_lambda': 0.8,
    'verbose': 10,
    #'is_unbalance': True
}
 
gbm_uo = lgb.train(params,
                lgb_train,
                #num_boost_round=100,
                #evals_result=metrics_lgb,
                verbose_eval=10,
                valid_sets=[lgb_train, lgb_eval],
                #early_stopping_rounds=5
                 )
 
#Predict
pred_train = gbm_uo.predict(train_early_x)
pred_valid = gbm_uo.predict(valid_early_x)
pred_test = gbm_uo.predict(test_x)
 
pred_train_binary = np.where(pred_train&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0.5,1,0)
pred_valid_binary = np.where(pred_valid&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0.5,1,0)
pred_test_binary = np.where(pred_test&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0.5,1,0)
 
#Check model performance
f1_score_train = f1_score(train_early_y, pred_train_binary)
f1_score_valid = f1_score(valid_early_y, pred_valid_binary)
f1_score_test = f1_score(test_y, pred_test_binary)
print('F1 train: {}'.format(f1_score_train))
print('F1 valid: {}'.format(f1_score_valid))
print('F1 test: {}'.format(f1_score_test))
 
confmat = confusion_matrix(test_y, pred_test_binary)
print(confmat)
 
roc_auc_score(test_y, pred_test_binary)
recall_score(test_y, pred_test_binary)
precision_score(test_y, pred_test_binary)
 
#Plot variable importances
print('Plotting feature importances...')
ax = lgb.plot_importance(gbm_uo, max_num_features=10)
plt.tight_layout()
plt.show()
 
plt.figure()
plt.hist(pred_test)
plt.show()
 
#----------------------Model 2: Under-Over prediction-------------------------
 
#Select features and create X and y
cols_to_use_gng = [#'venue_id',
                #'month',
                'hour',
               'localteam_id', 'visitorteam_id',
               #'winner', 'under_over',
               #'home_standing', 'away_standing',
               'standing_diff', 'points_diff',
               'goal_balance_home', 'goal_balance_away',
                 #'goal_balance_diff',
               'total_goals'
               ]
 
#Save features used by lgb to be called at prediction
np.save('cols_to_use_gng.npy', cols_to_use_gng)
 
df_final_gng = df[cols_to_use_gng]
ylabel_gng = df['goal_nogoal']
 
#Split in train and test
train_x, test_x, train_y, test_y = train_test_split(df_final_gng, ylabel_gng, shuffle=True)
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_x, train_y, shuffle=True)
 
lgb_train = lgb.Dataset(train_early_x, train_early_y)
lgb_eval = lgb.Dataset(valid_early_x, valid_early_y)
 
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    #'metric': {'auc'},
    'num_leaves': 16,
    'n_estimators': 100,
    'max_depth': -1,
    'learning_rate': 0.05,
    #'bagging_fraction': 0.9,
    #'reg_alpha': 0.8,
    #'reg_lambda': 0.8,
    'verbose': 10,
    #'is_unbalance': True
}
 
gbm_gng = lgb.train(params,
                lgb_train,
                #num_boost_round=100,
                #evals_result=metrics_lgb,
                verbose_eval=10,
                valid_sets=[lgb_train, lgb_eval],
                #early_stopping_rounds=5
                 )
 
#Predict
pred_train = gbm_gng.predict(train_early_x)
pred_valid = gbm_gng.predict(valid_early_x)
pred_test = gbm_gng.predict(test_x)
 
pred_train_binary = np.where(pred_train&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0.5,1,0)
pred_valid_binary = np.where(pred_valid&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0.5,1,0)
pred_test_binary = np.where(pred_test&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;0.5,1,0)
 
#Check model performance
f1_score_train = f1_score(train_early_y, pred_train_binary)
f1_score_valid = f1_score(valid_early_y, pred_valid_binary)
f1_score_test = f1_score(test_y, pred_test_binary)
print('F1 train: {}'.format(f1_score_train))
print('F1 valid: {}'.format(f1_score_valid))
print('F1 test: {}'.format(f1_score_test))
 
confmat = confusion_matrix(test_y, pred_test_binary)
print(confmat)
 
# #Plot variable importances
# print('Plotting feature importances...')
# ax = lgb.plot_importance(gbm_gng, max_num_features=10)
# plt.tight_layout()
# plt.show()
#
# plt.figure()
# plt.hist(pred_test)
# plt.show()
 
#----------------------Model 3: Game Winner prediction-------------------------
 
#Select features and create X and y
cols_to_use_winner = [#'venue_id',
                    #'month',
                    #'hour',
                    #'matchday',
                   'localteam_id', 'visitorteam_id',
                   #'winner', 'under_over',
                   'home_points_before_game', 'away_points_before_game',
                   'home_standing', 'away_standing',
                   'standing_diff', 'points_diff',
                   'goal_balance_home', 'goal_balance_away',
                   # 'goal_balance_diff',
                    #'total_goals',
                      'home_goals_for_before_game', 'away_goals_for_before_game',
                      'goals_for_diff', 'goals_against_diff'
                   ]
 
#Save features used by lgb to be called at prediction
#np.save('cols_to_use_winner.npy', cols_to_use_winner)
 
cat_feats = ['localteam_id', 'visitorteam_id']
 
df_final_winner = df[cols_to_use_winner]
ylabel_winner = df['winner']
 
#Split in train and test
train_x, test_x, train_y, test_y = train_test_split(df_final_winner, ylabel_winner)
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_x, train_y)
 
lgb_train = lgb.Dataset(train_early_x, train_early_y)
lgb_eval = lgb.Dataset(valid_early_x, valid_early_y)
 
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    "num_class" : 3,
    #'metric': { 'rmse'},
    'num_leaves': 16,
    'n_estimators': 50,
    'max_depth': 4,
    'learning_rate': 0.05,
    #'bagging_fraction': 0.9,
    #'reg_alpha': 0.8,
    #'reg_lambda': 0.8,
    #'verbose': 2,
    #'is_unbalance': [True]
}
 
gbm_winner = lgb.train(params,
                lgb_train,
                #num_boost_round=50,
                #evals_result=metrics_lgb,
                verbose_eval=10,
                valid_sets=[lgb_train, lgb_eval],
                categorical_feature= cat_feats,
                #early_stopping_rounds=5
                 )
 
pred_train = gbm_winner.predict(train_early_x)
pred_valid = gbm_winner.predict(valid_early_x)
pred_test = gbm_winner.predict(test_x)
pred_train_binary = np.argmax(pred_train, axis = 1)
pred_valid_binary = np.argmax(pred_valid, axis = 1)
pred_test_binary = np.argmax(pred_test, axis = 1)
 
f1_score_train = f1_score(train_early_y, pred_train_binary, average= 'macro')
f1_score_valid = f1_score(valid_early_y, pred_valid_binary, average= 'macro')
f1_score_test = f1_score(test_y, pred_test_binary, average= 'macro')
print('F1 train: {}'.format(f1_score_train))
print('F1 valid: {}'.format(f1_score_valid))
print('F1 test: {}'.format(f1_score_test))
 
confmat_train = confusion_matrix(train_early_y, pred_train_binary)
confmat_test = confusion_matrix(test_y, pred_test_binary)
print(confmat_train)
print(confmat_test)
 
# #Plot variable importances
# print('Plotting feature importances...')
# ax = lgb.plot_importance(gbm_winner, max_num_features=15)
# plt.tight_layout()
# plt.show()
 
#save models
gbm_uo.save_model('models/gbm_model_under_over_' + str(league_id) + '.txt')
gbm_gng.save_model('models/gbm_model_gng_' + str(league_id) + '.txt')
gbm_winner.save_model('models/gbm_model_game_result_' + str(league_id) + '.txt')&amp;amp;amp;amp;amp;amp;amp;amp;amp;lt;/pre&amp;amp;amp;amp;amp;amp;amp;amp;amp;gt;
