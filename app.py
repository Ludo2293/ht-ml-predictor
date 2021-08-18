from pychpp import CHPP
import pandas as pd
import numpy as np
from scipy.stats import norm
from joblib import load
from flask import Flask, request,  render_template
import os

# Connexion à l'API Hattrick
chpp = CHPP(os.getenv('consumer_key'),
            os.getenv('consumer_secret'),
            os.getenv('access_token_key'),
            os.getenv('access_token_secret'))

# Chargement du modèle de Machine Learning
reglog_mod=load('lgbm.joblib')

# Fonction pour surligner les surprises
def highlight_surprises(s, column):
    is_surp = pd.Series(data=False, index=s.index)
    is_surp[column] = s.loc[column] == 1
    return ['font-weight:bold; background-color:yellow; color:black' if is_surp.any() else '' for v in is_surp]


# Appli web
app = Flask(__name__)

# Template en cas d'erreur
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('index_err.html')

# PAGE D'ACCEUIL
@app.route('/', methods=("POST", "GET"))
def html_table():
    return render_template('index.html')

# Prédicteur de match individuel (organisation du code à revoir et probablement simplifier)
@app.route('/predict_match', methods=("POST", "GET"))
def html_predict():
    def calcul_pred(id_init):
        match = chpp.match(ht_id=id_init)
        # Estimation du Repli défensif
        diff_buts=match.home_team_goals-match.away_team_goals
        Pen_att_dom=1
        Bon_def_dom=1
        Pen_att_ext=1
        Bon_def_ext=1
        # Application de la fonction de "neutralisation" du repli défensif
        if diff_buts>=2:
            Pen_att_dom=.0008*diff_buts**2-.0419*diff_buts+1.0525
            Bon_def_dom=.0013*diff_buts**2+.0283*diff_buts+.9622
        elif diff_buts<=-2:
            Pen_att_ext=.0008*diff_buts**2+.0419*diff_buts+1.0525
            Bon_def_ext=.0013*diff_buts**2-.0283*diff_buts+.9622
            
        xG_dom=(match.home_team_rating_midfield==1)*(diff_buts==5)*5+(match.home_team_rating_midfield>1)*max(0.1,reglog_mod.predict([[match.home_team_rating_midfield**3/(match.home_team_rating_midfield**3+match.away_team_rating_midfield**3),
            .92*match.home_team_rating_right_att**3.5/(match.home_team_rating_right_att**3.5+match.away_team_rating_left_def**3.5),
            .92*match.home_team_rating_left_att**3.5/(match.home_team_rating_left_att**3.5+match.away_team_rating_right_def**3.5),
            .92*match.home_team_rating_mid_att**3.5/(match.home_team_rating_mid_att**3.5+match.away_team_rating_mid_def**3.5),
            .92*match.home_team_rating_ind_set_pieces_att**3.5/(match.home_team_rating_ind_set_pieces_att**3.5+match.away_team_rating_ind_set_pieces_def**3.5),
            1*(match.home_team_tactic_type=='1')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='2')*match.home_team_tactic_skill,
            1*(match.home_team_tactic_type=='3')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='4')*match.home_team_tactic_skill,
            1*(match.home_team_tactic_type=='7')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='8')*match.home_team_tactic_skill,
            1*(match.away_team_tactic_type=='1')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='7')*match.away_team_tactic_skill]])[0])
        # Extérieur
        xG_ext=(match.away_team_rating_midfield==1)*(diff_buts==-5)*5+(match.away_team_rating_midfield>1)*max(0.1,reglog_mod.predict([[match.away_team_rating_midfield**3/(match.home_team_rating_midfield**3+match.away_team_rating_midfield**3),
            .92*match.away_team_rating_right_att**3.5/(match.home_team_rating_right_att**3.5+match.away_team_rating_left_def**3.5),
            .92*match.away_team_rating_left_att**3.5/(match.home_team_rating_left_att**3.5+match.away_team_rating_right_def**3.5),
            .92*match.away_team_rating_mid_att**3.5/(match.home_team_rating_mid_att**3.5+match.away_team_rating_mid_def**3.5),
            .92*match.away_team_rating_ind_set_pieces_att**3.5/(match.home_team_rating_ind_set_pieces_att**3.5+match.away_team_rating_ind_set_pieces_def**3.5),
            1*(match.away_team_tactic_type=='1')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='2')*match.away_team_tactic_skill,
            1*(match.away_team_tactic_type=='3')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='4')*match.away_team_tactic_skill,
            1*(match.away_team_tactic_type=='7')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='8')*match.away_team_tactic_skill,
            1*(match.home_team_tactic_type=='1')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='7')*match.home_team_tactic_skill]])[0])
        
        # Calcul des probabilités de victoire
        Liste_matchs=pd.DataFrame(columns=['Home Team','Away Team','Score','xG Home','xG Away','Home win','Draw','Away win'])
        Tab_probas=pd.DataFrame(columns=range(0,11))
        Tab_probas.loc[0,0]=norm.cdf(0.5,xG_dom,1.2)*norm.cdf(0.5,xG_ext,1.2)
        for l in range(1,11):
            Tab_probas.loc[0,l]=norm.cdf(0.5,xG_dom,1.2)*(norm.cdf(l+0.5,xG_ext,1.2)-norm.cdf(l-0.5,xG_ext,1.2))
            for k in range(1,11):
                Tab_probas.loc[k,l]=(norm.cdf(k+0.5,xG_dom,1.2)-norm.cdf(k-0.5,xG_dom,1.2))*(norm.cdf(l+0.5,xG_ext,1.2)-norm.cdf(l-0.5,xG_ext,1.2))
        for k in range(1,11):
            Tab_probas.loc[k,0]=(norm.cdf(k+0.5,xG_dom,1.2)-norm.cdf(k-0.5,xG_dom,1.2))*norm.cdf(0.5,xG_ext,1.2)
        Proba1=0
        ProbaN=0
        Proba2=0
        for k in range(0,11):
            for l in range(0,11):
                if k>l:
                    Proba1=Proba1+Tab_probas.loc[k,l]
                elif k<l:
                    Proba2=Proba2+Tab_probas.loc[k,l]
                elif k==l:
                    ProbaN=ProbaN+Tab_probas.loc[k,l]
        Liste_matchs.loc[0]=[match.home_team_name,match.away_team_name,str(match.home_team_goals)+"-"+str(match.away_team_goals),
            xG_dom,xG_ext,str(round(Proba1*100,1))+"%",str(round(ProbaN*100,1))+"%",str(round(Proba2*100,1))+"%"]
    
        # Surprise ou non
        Liste_matchs['Surprise']=0
        Liste_matchs.loc[((Liste_matchs['Score'].str.split('-').str[0].astype(int))>=Liste_matchs['Score'].str.split('-').str[1].astype(int))
                         & (Liste_matchs['Away win'].str.split('%').str[0].astype(float)>=50),'Surprise']=1
        Liste_matchs.loc[((Liste_matchs['Score'].str.split('-').str[0].astype(int))<=Liste_matchs['Score'].str.split('-').str[1].astype(int))
                         & (Liste_matchs['Home win'].str.split('%').str[0].astype(float)>=50),'Surprise']=1
        Liste_matchs=Liste_matchs.style.hide_index().hide_columns(['Surprise']).set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}]).apply(highlight_surprises,column=['Surprise'],axis=1).set_precision(2).render()
        return Liste_matchs
    Liste_matchs=calcul_pred(int(request.form['MatchID']))
    return render_template('match_pred.html',tables=[Liste_matchs],titles=['MATCH RESULT PROBABILITIES'])
    
# PREDICTEUR DE LIGUE
@app.route('/predict_league', methods=("POST", "GET"))
def html_predict_league():
    def calcul_pred_league(id_league,num_saison):
        if num_saison==int(chpp.league_fixtures(ht_id=id_league).season):
            nb_matchs=min(4*int(chpp.league(ht_id=id_league).current_match_round)-4,56)
        else:
            nb_matchs=56
        liste_matchs=[chpp.match(ht_id=o.ht_id) for o in chpp.league_fixtures(ht_id=id_league,season=num_saison).matches][:nb_matchs]
        diff_buts=np.array([o.home_team_goals-o.away_team_goals for o in liste_matchs])
        # Si dernière journée ne championnat ou finale de coupe, pas de repli défensif
        Pen_att_dom=(diff_buts<2)+(.0008*diff_buts**2-.0419*diff_buts+1.0525)*(diff_buts>=2)
        Bon_def_dom=(diff_buts<2)+(.0013*diff_buts**2+.0283*diff_buts+.9622)*(diff_buts>=2)
        Pen_att_ext=(diff_buts>-2)+(.0008*diff_buts**2+.0419*diff_buts+1.0525)*(diff_buts<=-2)
        Bon_def_ext=(diff_buts>-2)+(.0013*diff_buts**2-.0283*diff_buts+.9622)*(diff_buts<=-2)
        home_team_rating_midfield=np.array([max(1,o.home_team_rating_midfield) for o in liste_matchs])
        away_team_rating_midfield=np.array([max(1,o.away_team_rating_midfield) for o in liste_matchs])
        home_team_rating_right_att=np.array([max(1,o.home_team_rating_right_att) for o in liste_matchs])/Pen_att_dom
        home_team_rating_left_att=np.array([max(1,o.home_team_rating_left_att) for o in liste_matchs])/Pen_att_dom
        home_team_rating_mid_att=np.array([max(1,o.home_team_rating_mid_att) for o in liste_matchs])/Pen_att_dom
        away_team_rating_right_att=np.array([max(1,o.away_team_rating_right_att) for o in liste_matchs])/Pen_att_ext
        away_team_rating_left_att=np.array([max(1,o.away_team_rating_left_att) for o in liste_matchs])/Pen_att_ext
        away_team_rating_mid_att=np.array([max(1,o.away_team_rating_mid_att) for o in liste_matchs])/Pen_att_ext
        home_team_rating_right_def=np.array([max(1,o.home_team_rating_right_def) for o in liste_matchs])/Bon_def_dom
        home_team_rating_left_def=np.array([max(1,o.home_team_rating_left_def) for o in liste_matchs])/Bon_def_dom
        home_team_rating_mid_def=np.array([max(1,o.home_team_rating_mid_def) for o in liste_matchs])/Bon_def_dom
        away_team_rating_right_def=np.array([max(1,o.away_team_rating_right_def) for o in liste_matchs])/Bon_def_ext
        away_team_rating_left_def=np.array([max(1,o.away_team_rating_left_def) for o in liste_matchs])/Bon_def_ext
        away_team_rating_mid_def=np.array([max(1,o.away_team_rating_mid_def) for o in liste_matchs])/Bon_def_ext
        home_team_rating_ind_set_pieces_att=np.array([max(1,o.home_team_rating_ind_set_pieces_att) for o in liste_matchs])
        home_team_rating_ind_set_pieces_def=np.array([max(1,o.home_team_rating_ind_set_pieces_def) for o in liste_matchs])
        away_team_rating_ind_set_pieces_att=np.array([max(1,o.away_team_rating_ind_set_pieces_att) for o in liste_matchs])
        away_team_rating_ind_set_pieces_def=np.array([max(1,o.away_team_rating_ind_set_pieces_def) for o in liste_matchs])
        home_team_tactic_skill_1=np.array([1*(o.home_team_tactic_type=='1')*o.home_team_tactic_skill for o in liste_matchs])
        home_team_tactic_skill_2=np.array([1*(o.home_team_tactic_type=='2')*o.home_team_tactic_skill for o in liste_matchs])
        home_team_tactic_skill_3=np.array([1*(o.home_team_tactic_type=='3')*o.home_team_tactic_skill for o in liste_matchs])
        home_team_tactic_skill_4=np.array([1*(o.home_team_tactic_type=='4')*o.home_team_tactic_skill for o in liste_matchs])
        home_team_tactic_skill_7=np.array([1*(o.home_team_tactic_type=='7')*o.home_team_tactic_skill for o in liste_matchs])
        home_team_tactic_skill_8=np.array([1*(o.home_team_tactic_type=='8')*o.home_team_tactic_skill for o in liste_matchs])
        away_team_tactic_skill_1=np.array([1*(o.away_team_tactic_type=='1')*o.away_team_tactic_skill for o in liste_matchs])
        away_team_tactic_skill_2=np.array([1*(o.away_team_tactic_type=='2')*o.away_team_tactic_skill for o in liste_matchs])
        away_team_tactic_skill_3=np.array([1*(o.away_team_tactic_type=='3')*o.away_team_tactic_skill for o in liste_matchs])
        away_team_tactic_skill_4=np.array([1*(o.away_team_tactic_type=='4')*o.away_team_tactic_skill for o in liste_matchs])
        away_team_tactic_skill_7=np.array([1*(o.away_team_tactic_type=='7')*o.away_team_tactic_skill for o in liste_matchs])
        away_team_tactic_skill_8=np.array([1*(o.away_team_tactic_type=='8')*o.away_team_tactic_skill for o in liste_matchs])
        
        # A retester dès que possible avec le traitement des forfaits
        xG_dom=(home_team_rating_midfield==1)*(diff_buts==5)*5+(home_team_rating_midfield>1)*np.around(reglog_mod.predict(pd.concat([pd.DataFrame(home_team_rating_midfield**3/(home_team_rating_midfield**3+away_team_rating_midfield**3)),
            pd.DataFrame(.92*(home_team_rating_right_att)**3.5/(home_team_rating_right_att**3.5+(away_team_rating_left_def)**3.5)),
            pd.DataFrame(.92*(home_team_rating_left_att)**3.5/(home_team_rating_left_att**3.5+(away_team_rating_right_def)**3.5)),
            pd.DataFrame(.92*(home_team_rating_mid_att)**3.5/(home_team_rating_mid_att**3.5+(away_team_rating_mid_def)**3.5)),
            pd.DataFrame(.92*home_team_rating_ind_set_pieces_att**3.5/(home_team_rating_ind_set_pieces_att**3.5+away_team_rating_ind_set_pieces_def**3.5)),
            pd.DataFrame(home_team_tactic_skill_1),pd.DataFrame(home_team_tactic_skill_2),pd.DataFrame(home_team_tactic_skill_3),
            pd.DataFrame(home_team_tactic_skill_4),pd.DataFrame(home_team_tactic_skill_7),pd.DataFrame(home_team_tactic_skill_8),
            pd.DataFrame(away_team_tactic_skill_1),pd.DataFrame(away_team_tactic_skill_7)],axis=1),
            num_iteration=reglog_mod.best_iteration_),decimals=2)
        # On définit une prévision "plancher" à 0.1 (hors cas de forfait)
        xG_dom[(xG_dom<.1) & (home_team_rating_midfield!=1)]=.1
        
        xG_ext=(home_team_rating_midfield==1)*(diff_buts==-5)*5+(home_team_rating_midfield>1)*np.around(reglog_mod.predict(pd.concat([pd.DataFrame(away_team_rating_midfield**3/(away_team_rating_midfield**3+home_team_rating_midfield**3)),
            pd.DataFrame(.92*(away_team_rating_right_att)**3.5/(away_team_rating_right_att**3.5+(home_team_rating_left_def)**3.5)),
            pd.DataFrame(.92*(away_team_rating_left_att)**3.5/(away_team_rating_left_att**3.5+(home_team_rating_right_def)**3.5)),
            pd.DataFrame(.92*(away_team_rating_mid_att)**3.5/(away_team_rating_mid_att**3.5+(home_team_rating_mid_def)**3.5)),
            pd.DataFrame(.92*away_team_rating_ind_set_pieces_att**3.5/(away_team_rating_ind_set_pieces_att**3.5+home_team_rating_ind_set_pieces_def**3.5)),
            pd.DataFrame(away_team_tactic_skill_1),pd.DataFrame(away_team_tactic_skill_2),pd.DataFrame(away_team_tactic_skill_3),
            pd.DataFrame(away_team_tactic_skill_4),pd.DataFrame(away_team_tactic_skill_7),pd.DataFrame(away_team_tactic_skill_8),
            pd.DataFrame(home_team_tactic_skill_1),pd.DataFrame(home_team_tactic_skill_7)],axis=1),
            num_iteration=reglog_mod.best_iteration_),decimals=2)
        xG_ext[(xG_ext<.1) & (away_team_rating_midfield!=1)]=.1
        
        
        Liste_matchs=pd.DataFrame(columns=['Home Team','Away Team','Score','xG Home','xG Away','Home win','Draw','Away win','Xpts Home','Xpts Away','Rpts Home','Rpts Away'])
        for i in range(0,nb_matchs):
            Tab_probas=pd.DataFrame(columns=range(0,11))
            Tab_probas.loc[0,0]=norm.cdf(0.5,xG_dom[i],1)*norm.cdf(0.5,xG_ext[i],1)
            for l in range(1,11):
                Tab_probas.loc[0,l]=norm.cdf(0.5,xG_dom[i],1)*(norm.cdf(l+0.5,xG_ext[i],1)-norm.cdf(l-0.5,xG_ext[i],1))
                for k in range(1,11):
                    Tab_probas.loc[k,l]=(norm.cdf(k+0.5,xG_dom[i],1)-norm.cdf(k-0.5,xG_dom[i],1))*(norm.cdf(l+0.5,xG_ext[i],1)
                                        -norm.cdf(l-0.5,xG_ext[i],1))
            for k in range(1,11):
                Tab_probas.loc[k,0]=(norm.cdf(k+0.5,xG_dom[i],1)-norm.cdf(k-0.5,xG_dom[i],1))*norm.cdf(0.5,xG_ext[i],1)
            Proba1=0
            ProbaN=0
            Proba2=0
            for k in range(0,11):
                for l in range(0,11):
                    if k>l:
                        Proba1=Proba1+Tab_probas.loc[k,l]
                    elif k<l:
                        Proba2=Proba2+Tab_probas.loc[k,l]
                    elif k==l:
                        ProbaN=ProbaN+Tab_probas.loc[k,l]
            Liste_matchs.loc[i]=[liste_matchs[i].home_team_name,liste_matchs[i].away_team_name,
                str(liste_matchs[i].home_team_goals)+"-"+str(liste_matchs[i].away_team_goals),xG_dom[i],
                xG_ext[i],str(round(Proba1*100,1))+"%",str(round(ProbaN*100,1))+"%",
                str(round(Proba2*100,1))+"%",round(3*Proba1+ProbaN,2),round(3*Proba2+ProbaN,2),
                3*(liste_matchs[i].home_team_goals>liste_matchs[i].away_team_goals)+
                    1*(liste_matchs[i].home_team_goals==liste_matchs[i].away_team_goals),
                3*(liste_matchs[i].away_team_goals>liste_matchs[i].home_team_goals)+
                    1*(liste_matchs[i].home_team_goals==liste_matchs[i].away_team_goals)]
        
        
        classement=pd.DataFrame(Liste_matchs.groupby('Home Team').agg({'Xpts Home':'sum','Rpts Home':'sum'}).reset_index()[['Home Team','Xpts Home','Rpts Home']]).merge(pd.DataFrame(Liste_matchs.groupby('Away Team').agg({'Xpts Away':'sum','Rpts Away':'sum'}).reset_index()[['Away Team','Xpts Away','Rpts Away']]).rename({'Away Team':'Home Team'},axis=1),on='Home Team',how='left').fillna(0)
        classement['Real points']=classement['Rpts Home']+classement['Rpts Away']
        classement['Expected points']=classement['Xpts Home']+classement['Xpts Away']
        classement['Points difference']=classement['Real points']-classement['Expected points']
        classement=classement.rename({'Home Team':'Team'},axis=1).drop(['Xpts Home','Xpts Away','Rpts Home','Rpts Away'],axis=1).sort_values(by='Real points',ascending=0)
        classement=classement.sort_values(by='Expected points',ascending=0)
        Liste_matchs=Liste_matchs.drop(['Xpts Home','Xpts Away','Rpts Home','Rpts Away'],axis=1)
            
        # Surprise ou non
        Liste_matchs['Surprise']=0
        Liste_matchs.loc[((Liste_matchs['Score'].str.split('-').str[0].astype(int))>=Liste_matchs['Score'].str.split('-').str[1].astype(int))
            & (Liste_matchs['Away win'].str.split('%').str[0].astype(float)>=50),'Surprise']=1
        Liste_matchs.loc[((Liste_matchs['Score'].str.split('-').str[0].astype(int))<=Liste_matchs['Score'].str.split('-').str[1].astype(int))
            & (Liste_matchs['Home win'].str.split('%').str[0].astype(float)>=50),'Surprise']=1
        classement=classement.style.hide_index().set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}]).format({'Points difference':'{:.1f}','Expected points':'{:.1f}','Real points':'{:.0f}'})\
            .apply(lambda x:['font-weight:bold; background-color:limegreen' if value>=1 else '' for value in x],subset=['Points difference'])\
            .apply(lambda x:['font-weight:bold; background-color:green' if value>=3 else '' for value in x],subset=['Points difference'])\
            .apply(lambda x:['font-weight:bold; background-color:lightcoral' if value<=-1 else '' for value in x],subset=['Points difference'])\
            .apply(lambda x:['font-weight:bold; background-color:red' if value<=-3 else '' for value in x],subset=['Points difference']).render()
        Liste_matchs=Liste_matchs.style.hide_index().hide_columns(['Surprise']).set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
           {"selector": "td", "props": [("text-align", "center")]}]).apply(highlight_surprises,column=['Surprise'],axis=1).set_precision(2).render()
        
        return Liste_matchs, classement
    Liste_matchs, classement=calcul_pred_league(int(request.form['LeagueID']),int(request.form['Season']))
    return render_template('league_pred.html',tables=[classement,Liste_matchs],titles=['SIMULATED RANKING - What is the theorical ranking ?','COMPLETE MATCH LIST - Surprising results are highlighted in yellow'])

if __name__ == '__main__':
    app.run()