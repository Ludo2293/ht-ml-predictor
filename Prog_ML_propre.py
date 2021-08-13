## Import des packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
from scipy.stats import norm
from joblib import dump,load
from sklearn.model_selection import GridSearchCV

#### DICO DONNEES
# TacticType : 0 - Normal
# TacticType : 1 - Pressing
# TacticType : 2 - Contre-attaques
# TacticType : 3 - Attaque au centre
# TacticType : 4 - Attaque sur les ailes
# TacticType : 7 - Jeu créatif
# TacticType : 8 - Tirs lointains


# Import des données de joueurs : Nombre de joueurs à spé dans chaque équipe et par poste
# Référentiels des spés et des postes
spes=pd.read_csv('playerperm.csv')[['PlayerID','Spec']].drop_duplicates()
roles=pd.read_csv('role.csv')[['RoleID','RoleType']].drop_duplicates()
df=pd.DataFrame()
for i in range(76,78):
    exec("lineup_"+str(i)+"=pd.read_csv('lineup_"+str(i)+".csv')")
    exec("lineup_"+str(i)+"=pd.read_csv('lineup_"+str(i)+".csv')")
    # Liste des matchs à enlever : ceux pour lesquels tous les joueurs ne sont pas dans le référentiel, et qui pourraient donc
    # avoir des spécialités sans qu'on ne le prenne en compte
    exec("matchs_a_virer=lineup_"+str(i)+"[lineup_"+str(i)+"['PlayerID'].isin(set(lineup_"+str(i)+"['PlayerID']).difference(spes['PlayerID']))]['MatchID'].unique()")
    exec("lineup_"+str(i)+"=lineup_"+str(i)+"[~lineup_"+str(i)+"['MatchID'].isin(matchs_a_virer)]")
    exec("lineup_"+str(i)+"=lineup_"+str(i)+"[~lineup_"+str(i)+"['RoleID'].isin([100,17])].merge(spes[spes['Spec']!=0],on='PlayerID',how='inner').merge(roles,on='RoleID',how='left')")
    # Prise en compte des remplacements et changements de position
    exec("lineup_"+str(i)+"=lineup_"+str(i)+".sort_values(by=['MatchID','PlayerID','RoleID','MinuteEnter'],ascending=True)")
    exec("lineup_"+str(i)+"['MinuteSortie']=90")
    exec("lineup_"+str(i)+".loc[lineup_"+str(i)+"['PlayerID']==lineup_"+str(i)+"['PlayerID'].shift(-1),'MinuteSortie']=lineup_"+str(i)+"['MinuteEnter'].shift(-1)")
    exec("lineup_"+str(i)+"['MinuteEnter']=(lineup_"+str(i)+"['MinuteSortie']-lineup_"+str(i)+"['MinuteEnter'])/90")
    # Suppression des joueurs remplacés
    exec("lineup_"+str(i)+"=lineup_"+str(i)+"[lineup_"+str(i)+"['Behaviour']!=888]")
    exec("tab_spes=pd.DataFrame(lineup_"+str(i)+".groupby(['MatchID','TeamID','RoleType','Spec']).agg({'MinuteEnter':'sum'})).reset_index()")
    tab_spes['IM_Tech']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='IM')*(tab_spes['Spec']==1)
    tab_spes['IM_Rapide']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='IM')*(tab_spes['Spec']==2)
    tab_spes['IM_Costaud']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='IM')*(tab_spes['Spec']==3)
    tab_spes['IM_Impr']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='IM')*(tab_spes['Spec']==4)
    tab_spes['IM_Head']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='IM')*(tab_spes['Spec']==5)
    tab_spes['CD_Tech']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='CD')*(tab_spes['Spec']==1)
    tab_spes['CD_Rapide']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='CD')*(tab_spes['Spec']==2)
    tab_spes['CD_Costaud']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='CD')*(tab_spes['Spec']==3)
    tab_spes['CD_Impr']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='CD')*(tab_spes['Spec']==4)
    tab_spes['CD_Head']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='CD')*(tab_spes['Spec']==5)
    tab_spes['WB_Tech']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='WB')*(tab_spes['Spec']==1)
    tab_spes['WB_Rapide']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='WB')*(tab_spes['Spec']==2)
    tab_spes['WB_Costaud']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='WB')*(tab_spes['Spec']==3)
    tab_spes['WB_Impr']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='WB')*(tab_spes['Spec']==4)
    tab_spes['WB_Head']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='WB')*(tab_spes['Spec']==5)
    tab_spes['FW_Tech']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='FW')*(tab_spes['Spec']==1)
    tab_spes['FW_Rapide']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='FW')*(tab_spes['Spec']==2)
    tab_spes['FW_Costaud']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='FW')*(tab_spes['Spec']==3)
    tab_spes['FW_Impr']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='FW')*(tab_spes['Spec']==4)
    tab_spes['FW_Head']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='FW')*(tab_spes['Spec']==5)
    tab_spes['W_Tech']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='W')*(tab_spes['Spec']==1)
    tab_spes['W_Rapide']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='W')*(tab_spes['Spec']==2)
    tab_spes['W_Costaud']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='W')*(tab_spes['Spec']==3)
    tab_spes['W_Impr']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='W')*(tab_spes['Spec']==4)
    tab_spes['W_Head']=tab_spes['MinuteEnter']*(tab_spes['RoleType']=='W')*(tab_spes['Spec']==5)
    tab_spes=tab_spes.groupby(['MatchID','TeamID']).agg({'IM_Tech':'sum','IM_Rapide':'sum','IM_Costaud':'sum',
        'IM_Impr':'sum','IM_Head':'sum','CD_Tech':'sum','CD_Rapide':'sum','CD_Costaud':'sum',
        'CD_Impr':'sum','CD_Head':'sum','WB_Tech':'sum','WB_Rapide':'sum','WB_Costaud':'sum',
        'WB_Impr':'sum','WB_Head':'sum','FW_Tech':'sum','FW_Rapide':'sum','FW_Costaud':'sum',
        'FW_Impr':'sum','FW_Head':'sum','W_Tech':'sum','W_Rapide':'sum','W_Costaud':'sum',
        'W_Impr':'sum','W_Head':'sum',}).reset_index()
    exec("del lineup_"+str(i))
    # Import de la base des matchs
    exec("df_"+str(i)+"=pd.read_csv('matchteam_"+str(i)+".csv').drop_duplicates().drop(['Location', \
         'Formation','PossH1','PossH2'],axis=1)")
    # Suppression des matchs avec un forfait
    exec("df_"+str(i)+"=df_"+str(i)+"[df_"+str(i)+"['RatingMidfield']>0]")
    # On écarte les matchs avec prolongation
    exec("match_"+str(i)+"=pd.read_csv('match_"+str(i)+".csv',low_memory=False).drop_duplicates()")
    exec("df_"+str(i)+"=df_"+str(i)+".merge(match_"+str(i)+"[match_"+str(i)+"['MatchLengthID']==1]['MatchID'],on='MatchID',how='inner').drop_duplicates()")
    # Ajout des données de spécialités
    exec("df_"+str(i)+"=df_"+str(i)+".merge(tab_spes,on=['MatchID','TeamID'],how='left').fillna(0)")
    exec("df=pd.concat([df,df_"+str(i)+"],axis=0)")
    exec("del df_"+str(i)+",match_"+str(i)+",tab_spes")

del roles,spes

# Estimation de la baisse de notes due au repli défensif, puis correction
df1=df[['MatchID','Opp_TeamID','Goals']].rename({'Opp_TeamID':'TeamID','Goals':'Goals_adv'},axis=1)
df=df.merge(df1,on=['MatchID','TeamID'],how='left')
del df1
df['Goals_diff']=df['Goals']-df['Goals_adv']
df['Pen_att']=1
df['Bon_def']=1
df.loc[(df['Goals']-df['Goals_adv'])>2,'Pen_att']=.0008*(df['Goals']-df['Goals_adv'])**2-.0419*(df['Goals']-df['Goals_adv'])+1.0525
df.loc[(df['Goals']-df['Goals_adv'])>2,'Bon_def']=.0013*(df['Goals']-df['Goals_adv'])**2+.0283*(df['Goals']-df['Goals_adv'])+.9622
for i in ['RatingRightAtt','RatingLeftAtt','RatingMidAtt']:
    df[i]=df[i]/df['Pen_att']
for i in ['RatingRightDef','RatingLeftDef','RatingMidDef']:
    df[i]=df[i]/df['Bon_def']
  
           
### On récupère les notes de l'adversaire
# Retraitements des notes adverses
df1=df.drop(['TeamID','Goals','Goals_adv','Goals_diff','Pen_att','Bon_def'],
    axis=1).rename({'RatingMidfield':'RatingMidfield_adv',
    'RatingRightDef':'RatingRightDef_adv','RatingMidDef':'RatingMidDef_adv',
    'RatingLeftDef':'RatingLeftDef_adv','RatingRightAtt':'RatingRightAtt_adv',
    'RatingMidAtt':'RatingMidAtt_adv','RatingLeftAtt':'RatingLeftAtt_adv',
    'RatingISPDef':'RatingISPDef_adv','RatingISPAtt':'RatingISPAtt_adv',
    'TacticSkill':'TacticSkill_adv','TacticType':'TacticType_adv','IM_Tech':'IM_Tech_adv',
    'IM_Rapide':'IM_Rapide_adv','IM_Costaud':'IM_Costaud_adv','IM_Impr':'IM_Impr_adv','IM_Head':'IM_Head_adv',
    'CD_Tech':'CD_Tech_adv','CD_Rapide':'CD_Rapide_adv','CD_Costaud':'CD_Costaud_adv','CD_Impr':'CD_Impr_adv',
    'CD_Head':'CD_Head_adv','WB_Tech':'WB_Tech_adv','WB_Rapide':'WB_Rapide_adv','WB_Costaud':'WB_Costaud_adv',
    'WB_Impr':'WB_Impr_adv','WB_Head':'WB_Head_adv','FW_Tech':'FW_Tech_adv','FW_Rapide':'FW_Rapide_adv',
    'FW_Costaud':'FW_Costaud_adv','FW_Impr':'FW_Impr_adv','FW_Head':'FW_Head_adv','W_Tech':'W_Tech_adv',
    'W_Rapide':'W_Rapide_adv','W_Costaud':'W_Costaud_adv','W_Impr':'W_Impr_adv','W_Head':'W_Head_adv',
    'Opp_TeamID':'TeamID'},axis=1)
                    
# Ajout des notes adverses                    
df=df.merge(df1,on=['MatchID','TeamID'],how='left').drop(['Opp_TeamID','Pen_att','Bon_def','Goals_adv','Goals_diff'],axis=1)
del df1

# Création d'un ID unique
df['ID']=df['MatchID'].astype(str)+'_'+df['TeamID'].astype(str)
df['MidfieldRatio']=df['RatingMidfield']**3/(df['RatingMidfield']**3+df['RatingMidfield_adv']**3)
df['RightAttRatio']=.92*df['RatingRightAtt']**3.5/(df['RatingRightAtt']**3.5+df['RatingLeftDef_adv']**3.5)
df['LeftAttRatio']=.92*df['RatingLeftAtt']**3.5/(df['RatingLeftAtt']**3.5+df['RatingRightDef_adv']**3.5)
df['MidAttRatio']=.92*df['RatingMidAtt']**3.5/(df['RatingMidAtt']**3.5+df['RatingMidDef_adv']**3.5)
df['RightAtt_advRatio']=.92*df['RatingRightAtt_adv']**3.5/(df['RatingRightAtt_adv']**3.5+df['RatingLeftDef']**3.5)
df['LeftAtt_advRatio']=.92*df['RatingLeftAtt_adv']**3.5/(df['RatingLeftAtt_adv']**3.5+df['RatingRightDef']**3.5)
df['MidAtt_advRatio']=.92*df['RatingMidAtt_adv']**3.5/(df['RatingMidAtt_adv']**3.5+df['RatingMidDef']**3.5)
df['ISPAttRatio']=.92*df['RatingISPAtt']**3.5/(df['RatingISPAtt']**3.5+df['RatingISPDef_adv']**3.5)
df['ISPAtt_advRatio']=.92*df['RatingISPAtt_adv']**3.5/(df['RatingISPAtt_adv']**3.5+df['RatingISPDef']**3.5)
#df[df['RatingMidAtt']==df['RatingMidDef_adv']][['MidAttRatio','RatingMidAtt','RatingMidDef_adv']]

# Analyse des notes, pour voir s'il n'y a pas des matchs à virer...
df=df.drop(['RatingMidfield','RatingMidfield_adv','RatingRightAtt',
                'RatingRightAtt_adv','RatingMidAtt','RatingMidAtt_adv','RatingLeftAtt',
                'RatingLeftAtt_adv','RatingRightDef','RatingRightDef_adv',
                'RatingLeftDef','RatingLeftDef_adv','RatingMidDef','RatingMidDef_adv',
                'RatingISPAtt','RatingISPDef','RatingISPAtt_adv','RatingISPDef_adv'],axis=1)


# Split entre base de test et base de training
train, test=train_test_split(df,test_size=.2)
del df
train['train']=1
test['train']=0

#test2=test.copy()
# Les tables test et train sont concaténées afin que le target encoding puisse aussi se faire sur 
df = train.append(test)
del train
df.loc[df['train']==0,'Goals']=np.nan

# Traitement des variables catégorielles
# One-hot encoding
df=pd.concat([df,pd.get_dummies(df['TacticType'],prefix='TacticType'),pd.get_dummies(df['TacticType_adv'],prefix='TacticType_adv')],axis=1)
df=df.drop(['TacticType','TacticType_adv','TacticType_0','TacticType_adv_0'],axis=1)

df['TacticSkill_1']=df['TacticSkill']*df['TacticType_1']
df['TacticSkill_2']=df['TacticSkill']*df['TacticType_2']
df['TacticSkill_3']=df['TacticSkill']*df['TacticType_3']
df['TacticSkill_4']=df['TacticSkill']*df['TacticType_4']
df['TacticSkill_7']=df['TacticSkill']*df['TacticType_7']
df['TacticSkill_8']=df['TacticSkill']*df['TacticType_8']
df['TacticSkill_adv_1']=df['TacticSkill_adv']*df['TacticType_adv_1']
df['TacticSkill_adv_2']=df['TacticSkill_adv']*df['TacticType_adv_2']
df['TacticSkill_adv_3']=df['TacticSkill_adv']*df['TacticType_adv_3']
df['TacticSkill_adv_4']=df['TacticSkill_adv']*df['TacticType_adv_4']
df['TacticSkill_adv_7']=df['TacticSkill_adv']*df['TacticType_adv_7']
df['TacticSkill_adv_8']=df['TacticSkill_adv']*df['TacticType_adv_8']

df2=df.drop(['W_Rapide_adv','IM_Tech_adv','FW_Tech','FW_Head_adv',
             'IM_Tech','W_Head','IM_Rapide','CD_Head_adv','IM_Impr','FW_Tech_adv',
             'FW_Rapide_adv','FW_Costaud','WB_Head','FW_Impr','IM_Costaud',
             'FW_Impr_adv','IM_Impr_adv','IM_Rapide_adv',
             'IM_Costaud_adv','W_Head_adv','W_Impr',
             'CD_Costaud_adv','WB_Head_adv',
             'WB_Rapide','FW_Costaud_adv','CD_Impr','W_Impr_adv','WB_Rapide_adv',
             'CD_Impr_adv','CD_Rapide_adv','CD_Costaud','W_Tech_adv','W_Tech','WB_Impr',
             'WB_Costaud','CD_Tech_adv','W_Costaud_adv','W_Costaud','CD_Rapide',
             'WB_Costaud_adv','CD_Tech','WB_Impr_adv','WB_Tech','TacticType_adv_2',
             'WB_Tech_adv','TacticType_adv_4','TacticType_adv_3',
             'TacticType_adv_8','TacticSkill','TacticSkill_adv',
             'TacticType_adv_1','TacticType_adv_7','TacticType_1',
             'TacticType_2','TacticType_3','TacticType_4','TacticType_7',
             'TacticType_8','IM_Head_adv','FW_Head','CD_Head','FW_Rapide','IM_Head',
             'W_Rapide','TacticSkill_adv_8'],axis=1)

# Stockage de l'intitulé des colonnes
dump(df2.columns,'colonnes.joblib')


# On ne conserve que les variables numériques, car seules elles pourront entrer dans le LGBM
dtypes = df2.drop(['MatchID','TeamID'],axis=1).dtypes.map(str)
numerical_features = list(dtypes[dtypes.isin(['int64','float64','float32','int32','uint8'])].index)
X_full = df2.drop(['MatchID','TeamID'],axis=1)[numerical_features].copy()

# Préparation de la base de modélisation
X = X_full[(X_full['train'] == 1)].copy()
X_test = X_full[X_full['train'] == 0].copy()
y = X_full[(X_full['train'] == 1)]['Goals']
del X_full
indexes=pd.DataFrame(df2[df2['train']==0]['ID'],columns={'ID'}).drop_duplicates()
val_reelles=indexes.merge(test[['ID','Goals']].drop_duplicates(),on='ID',how='left')
y_test=val_reelles['Goals']

X.drop(['Goals', 'train'], axis=1, inplace=True)
X_test.drop(['Goals', 'train'], axis=1, inplace=True)

# Tuning des hyperparamètres, à l'aide d'un grid Search
params = {
    'num_leaves': [200],
    'n_estimators': [900],
    #'max_depth':[-1],
    #'boosting_type':['gbdt'],
    #'subsample_for_bin':[200000],
    #'reg_alpha':[.01,.1]
    #'reg_lambda':[.01,.1]
}

grid = GridSearchCV(lgbm.LGBMRegressor(random_state=0), params, scoring='neg_root_mean_squared_error', cv=5, verbose=4)
grid.fit(X,y)
print(grid.best_estimator_)
print(grid.best_score_)
grid.cv_results_

# Lancement de l'algorithme Light GBM avec les hyperparamètres optimisés
reglog_mod=lgbm.LGBMRegressor(num_leaves=200,n_estimators=900,random_state=0) # RMSE : 1.07097
reglog_mod.fit(X, y,eval_set=[(X_test,y_test)],eval_metric='l2_root')
# Prédictions sur l'échantillon test
y_pred=reglog_mod.predict(X_test,num_iteration=reglog_mod.best_iteration_)
# Calcul du R² sur la base de test
accuracy = round(reglog_mod.score(X_test,y_test)*100,2)
print(accuracy)