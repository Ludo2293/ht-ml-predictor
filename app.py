from pychpp import CHPP
from pychpp.chpp import CHPPBase
import pandas as pd
import numpy as np
from scipy.stats import norm
from joblib import load
from flask import Flask, request,  render_template
import os
import re

print(dir(CHPP))
# Connexion à l'API Hattrick
chpp = CHPP(os.getenv('consumer_key'),
            os.getenv('consumer_secret'),
            os.getenv('access_token_key'),
            os.getenv('access_token_secret'))

# Chargement du modèle de Machine Learning
model=load('lgbm.joblib')

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
        try:
            match = chpp.match(ht_id=id_init,source="htointegrated")
        except:
            match = chpp.match(ht_id=id_init)
        # Estimation du Repli défensif
        diff_buts=match.home_team_goals-match.away_team_goals
        # Application de la fonction de "neutralisation" du repli défensif
        ## Calcul du nombre de minutes avec N buts d'écart
        liste_minutes=[]
        liste_hg=[]
        liste_ag=[]
        for i in range(0,len(match.goals)):
            liste_minutes.append(match.goals[i]['minute'])
            liste_hg.append(match.goals[i]['home_goals'])
            liste_ag.append(match.goals[i]['away_goals'])
        liste_minutes=[min(a,90) for a in liste_minutes]
        liste_db=[0]+[a-b for a,b in zip(liste_hg,liste_ag)]
        liste_db_att=[.91**(max(a,1)-1) for a in liste_db]
        liste_db_def=[1.075**(max(a,1)-1) for a in liste_db]
        liste_db_att_ext=[.91**-min(0,-max(-a,-1)+1) for a in liste_db]
        liste_db_def_ext=[1.075**-min(0,-max(-a,-1)+1) for a in liste_db]
        liste_md1=[0]+liste_minutes
        liste_md2=liste_minutes+[90]
        liste_md=[a-b for a,b in zip(liste_md2,liste_md1)]
        Pen_att_dom=sum(a*b for a,b in zip(liste_db_att,liste_md))/90
        Bon_def_dom=sum(a*b for a,b in zip(liste_db_def,liste_md))/90
        Pen_att_ext=sum(a*b for a,b in zip(liste_db_att_ext,liste_md))/90
        Bon_def_ext=sum(a*b for a,b in zip(liste_db_def_ext,liste_md))/90
        xG_dom=(match.home_team_rating_midfield==1)*(diff_buts==5)*5+(match.home_team_rating_midfield>1)*max(0.1,model.predict([[match.home_team_rating_midfield**3/(match.home_team_rating_midfield**3+match.away_team_rating_midfield**3),
            .92*(match.home_team_rating_right_att/Pen_att_dom)**3.5/((match.home_team_rating_right_att/Pen_att_dom)**3.5+(match.away_team_rating_left_def/Bon_def_ext)**3.5),
            .92*(match.home_team_rating_left_att/Pen_att_dom)**3.5/((match.home_team_rating_left_att/Pen_att_dom)**3.5+(match.away_team_rating_right_def/Bon_def_ext)**3.5),
            .92*(match.home_team_rating_mid_att/Pen_att_dom)**3.5/((match.home_team_rating_mid_att/Pen_att_dom)**3.5+(match.away_team_rating_mid_def/Bon_def_ext)**3.5),
            .92*match.home_team_rating_ind_set_pieces_att**3.5/(match.home_team_rating_ind_set_pieces_att**3.5+match.away_team_rating_ind_set_pieces_def**3.5),
            1*(match.home_team_tactic_type=='1')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='2')*match.home_team_tactic_skill,
            1*(match.home_team_tactic_type=='3')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='4')*match.home_team_tactic_skill,
            1*(match.home_team_tactic_type=='7')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='8')*match.home_team_tactic_skill,
            1*(match.away_team_tactic_type=='1')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='7')*match.away_team_tactic_skill]])[0])
        # Extérieur
        xG_ext=(match.away_team_rating_midfield==1)*(diff_buts==-5)*5+(match.away_team_rating_midfield>1)*max(0.1,model.predict([[match.away_team_rating_midfield**3/(match.home_team_rating_midfield**3+match.away_team_rating_midfield**3),
            .92*(match.away_team_rating_right_att/Pen_att_ext)**3.5/((match.away_team_rating_right_att/Pen_att_ext)**3.5+(match.home_team_rating_left_def/Bon_def_dom)**3.5),
            .92*(match.away_team_rating_left_att/Pen_att_ext)**3.5/((match.away_team_rating_left_att/Pen_att_ext)**3.5+(match.home_team_rating_right_def/Bon_def_dom)**3.5),
            .92*(match.away_team_rating_mid_att/Pen_att_ext)**3.5/((match.away_team_rating_mid_att/Pen_att_ext)**3.5+(match.home_team_rating_mid_def/Bon_def_dom)**3.5),
            .92*match.away_team_rating_ind_set_pieces_att**3.5/(match.away_team_rating_ind_set_pieces_att**3.5+match.home_team_rating_ind_set_pieces_def**3.5),
            1*(match.away_team_tactic_type=='1')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='2')*match.away_team_tactic_skill,
            1*(match.away_team_tactic_type=='3')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='4')*match.away_team_tactic_skill,
            1*(match.away_team_tactic_type=='7')*match.away_team_tactic_skill,1*(match.away_team_tactic_type=='8')*match.away_team_tactic_skill,
            1*(match.home_team_tactic_type=='1')*match.home_team_tactic_skill,1*(match.home_team_tactic_type=='7')*match.home_team_tactic_skill]])[0])
        
        # Calcul des probabilités de victoire
        Liste_matchs=pd.DataFrame(columns=['Home Team','Away Team','Score','xG Home','xG Away','Home win','Draw','Away win'])
        Tab_probas=pd.DataFrame(columns=range(0,15))
        Tab_probas.loc[0,0]=norm.cdf(0.5,(xG_dom>=.4)*(xG_dom<2)*(1.1314*xG_dom-.3065)+(xG_dom>=2)*xG_dom,
            .0052*xG_dom**3-.0957*xG_dom**2+.5098*xG_dom+.5666)*norm.cdf(0.5,(xG_ext>=.4)*(xG_ext<2)*(1.1314*xG_ext-.3065)+(xG_ext>=2)*xG_ext,.0052*xG_ext**3-.0957*xG_ext**2+.5098*xG_ext+.5666)
        for l in range(1,15):
            Tab_probas.loc[0,l]=norm.cdf(0.5,(xG_dom>=.4)*(xG_dom<2)*(1.1314*xG_dom-.3065)+(xG_dom>=2)*xG_dom,
                .0052*xG_dom**3-.0957*xG_dom**2+.5098*xG_dom+.5666)*(norm.cdf(l+0.5,(xG_ext>=.4)*(xG_ext<2)*(1.1314*xG_ext-.3065)+(xG_ext>=2)*xG_ext,.0052*xG_ext**3-.0957*xG_ext**2+.5098*xG_ext+.5666)
                -norm.cdf(l-0.5,(xG_ext>=.4)*(xG_ext<2)*(1.1314*xG_ext-.3065)+(xG_ext>=2)*xG_ext,.0052*xG_ext**3-.0957*xG_ext**2+.5098*xG_ext+.5666))
            for k in range(1,15):
                Tab_probas.loc[k,l]=(norm.cdf(k+0.5,(xG_dom>=.4)*(xG_dom<2)*(1.1314*xG_dom-.3065)+(xG_dom>=2)*xG_dom,.0052*xG_dom**3-.0957*xG_dom**2+.5098*xG_dom+.5666)
                -norm.cdf(k-0.5,(xG_dom>=.4)*(xG_dom<2)*(1.1314*xG_dom-.3065)+(xG_dom>=2)*xG_dom,.0052*xG_dom**3-.0957*xG_dom**2+.5098*xG_dom+.5666))*(norm.cdf(l+0.5,(xG_ext>=.4)*(xG_ext<2)*(1.1314*xG_ext-.3065)+(xG_ext>=2)*xG_ext,.0052*xG_ext**3-.0957*xG_ext**2+.5098*xG_ext+.5666)
                -norm.cdf(l-0.5,(xG_ext>=.4)*(xG_ext<2)*(1.1314*xG_ext-.3065)+(xG_ext>=2)*xG_ext,.0052*xG_ext**3-.0957*xG_ext**2+.5098*xG_ext+.5666))
        for k in range(1,15):
            Tab_probas.loc[k,0]=(norm.cdf(k+0.5,(xG_dom>=.4)*(xG_dom<2)*(1.1314*xG_dom-.3065)+(xG_dom>=2)*xG_dom,.0052*xG_dom**3-.0957*xG_dom**2+.5098*xG_dom+.5666)
            -norm.cdf(k-0.5,(xG_dom>=.4)*(xG_dom<2)*(1.1314*xG_dom-.3065)+(xG_dom>=2)*xG_dom,.0052*xG_dom**3-.0957*xG_dom**2+.5098*xG_dom+.5666))*norm.cdf(0.5,(xG_ext>=.4)*(xG_ext<2)*(1.1314*xG_ext-.3065)+(xG_ext>=2)*xG_ext,.0052*xG_ext**3-.0957*xG_ext**2+.5098*xG_ext+.5666)
        Proba1=0
        ProbaN=0
        Proba2=0
        for k in range(0,15):
            for l in range(0,15):
                if k>l:
                    Proba1=Proba1+Tab_probas.loc[k,l]
                elif k<l:
                    Proba2=Proba2+Tab_probas.loc[k,l]
                elif k==l:
                    ProbaN=ProbaN+Tab_probas.loc[k,l]
        try:
            Liste_matchs.loc[0]=[match.home_team_name,match.away_team_name,str(match.goals[len(match.goals)-1]['home_goals'])+"-"+str(match.goals[len(match.goals)-1]['away_goals']),
                xG_dom,xG_ext,str(int(round(Proba1*100,0)))+"%",str(100-(int(round(Proba1*100,0))+int(round(ProbaN*100,0))+int(round(Proba2*100,0)))+int(round(ProbaN*100,0)))+"%",str(int(round(Proba2*100,0)))+"%"]
        except:
            Liste_matchs.loc[0]=[match.home_team_name,match.away_team_name,"0-0",
                xG_dom,xG_ext,str(int(round(Proba1*100,0)))+"%",str(100-(int(round(Proba1*100,0))+int(round(ProbaN*100,0))+int(round(Proba2*100,0)))+int(round(ProbaN*100,0)))+"%",str(int(round(Proba2*100,0)))+"%"]
    
        # Surprise ou non
        Liste_matchs['Surprise']=0
        Liste_matchs.loc[((Liste_matchs['Score'].str.split('-').str[0].astype(int))>=Liste_matchs['Score'].str.split('-').str[1].astype(int))
                         & (Liste_matchs['Away win'].str.split('%').str[0].astype(float)>=50),'Surprise']=1
        Liste_matchs.loc[((Liste_matchs['Score'].str.split('-').str[0].astype(int))<=Liste_matchs['Score'].str.split('-').str[1].astype(int))
                         & (Liste_matchs['Home win'].str.split('%').str[0].astype(float)>=50),'Surprise']=1
        Liste_matchs=Liste_matchs.style.hide_index().hide_columns(['Surprise']).set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}]).apply(highlight_surprises,column=['Surprise'],axis=1).set_precision(2).render()
        liste_format_HT=re.sub('<style.*?</style>','',Liste_matchs,flags=re.DOTALL)
        liste_format_HT=re.sub('<table.*?>','<table>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<th.*?>','<th>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<td.*?>','<td>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('>\n.*?<','><',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<table>.*?<tr>','<table><tr>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<tr>.*?<th>','<tr><th>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('</th>.*?<th>','</th><th>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=liste_format_HT.replace('<tbody>','').replace('</tbody>','').replace('</thead>','').replace("<","[").replace(">","]")
        return Liste_matchs, liste_format_HT
    if request.method=='POST':
        Liste_matchs, liste_format_HT=calcul_pred(int(request.form['MatchID']))
    else:
        Liste_matchs, liste_format_HT=calcul_pred(int(request.args.get('MatchID')))
    return render_template('match_pred.html',tables=[Liste_matchs,liste_format_HT],titles=['MATCH RESULT PROBABILITIES','Copy the following code to share the result on Hattrick forum :'])
    
# PREDICTEUR DE LIGUE
@app.route('/predict_league', methods=("POST", "GET"))
def html_predict_league():
    def calcul_pred_league(id_league,num_saison):
        if num_saison==int(chpp.leaguefixtures(ht_id=id_league).season):
            nb_matchs=min(4*int(chpp.league(ht_id=id_league).current_match_round)-4,56)
        else:
            nb_matchs=56
        liste_matchs=[chpp.match(ht_id=o.ht_id) for o in chpp.leaguefixtures(ht_id=id_league,season=num_saison).matches][:nb_matchs]
        diff_buts=np.array([o.home_team_goals-o.away_team_goals for o in liste_matchs])
        # Repli défensif
        Pen_att_dom=[1 for x in range(0,nb_matchs)]
        Bon_def_dom=[1 for x in range(0,nb_matchs)]
        Pen_att_ext=[1 for x in range(0,nb_matchs)]
        Bon_def_ext=[1 for x in range(0,nb_matchs)]
        for a in range(0,nb_matchs):  
            match=liste_matchs[a]
            liste_minutes=[]
            liste_hg=[]
            liste_ag=[]
            for i in range(0,len(match.goals)):
                liste_minutes.append(match.goals[i]['minute'])
                liste_hg.append(match.goals[i]['home_goals'])
                liste_ag.append(match.goals[i]['away_goals'])
            liste_minutes=[min(a,90) for a in liste_minutes]
            liste_db=[0]+[a-b for a,b in zip(liste_hg,liste_ag)]
            liste_db_att=[.91**(max(a,1)-1) for a in liste_db]
            liste_db_def=[1.075**(max(a,1)-1) for a in liste_db]
            liste_db_att_ext=[.91**-min(0,-max(-a,-1)+1) for a in liste_db]
            liste_db_def_ext=[1.075**-min(0,-max(-a,-1)+1) for a in liste_db]
            liste_md1=[0]+liste_minutes
            liste_md2=liste_minutes+[90]
            liste_md=[a-b for a,b in zip(liste_md2,liste_md1)]
            Pen_att_dom[a]=sum(a*b for a,b in zip(liste_db_att,liste_md))/90
            Bon_def_dom[a]=sum(a*b for a,b in zip(liste_db_def,liste_md))/90
            Pen_att_ext[a]=sum(a*b for a,b in zip(liste_db_att_ext,liste_md))/90
            Bon_def_ext[a]=sum(a*b for a,b in zip(liste_db_def_ext,liste_md))/90
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
        xG_dom=(home_team_rating_midfield==1)*(diff_buts==5)*5+(home_team_rating_midfield>1)*np.around(model.predict(pd.concat([pd.DataFrame(home_team_rating_midfield**3/(home_team_rating_midfield**3+away_team_rating_midfield**3)),
            pd.DataFrame(.92*(home_team_rating_right_att)**3.5/(home_team_rating_right_att**3.5+(away_team_rating_left_def)**3.5)),
            pd.DataFrame(.92*(home_team_rating_left_att)**3.5/(home_team_rating_left_att**3.5+(away_team_rating_right_def)**3.5)),
            pd.DataFrame(.92*(home_team_rating_mid_att)**3.5/(home_team_rating_mid_att**3.5+(away_team_rating_mid_def)**3.5)),
            pd.DataFrame(.92*home_team_rating_ind_set_pieces_att**3.5/(home_team_rating_ind_set_pieces_att**3.5+away_team_rating_ind_set_pieces_def**3.5)),
            pd.DataFrame(home_team_tactic_skill_1),pd.DataFrame(home_team_tactic_skill_2),pd.DataFrame(home_team_tactic_skill_3),
            pd.DataFrame(home_team_tactic_skill_4),pd.DataFrame(home_team_tactic_skill_7),pd.DataFrame(home_team_tactic_skill_8),
            pd.DataFrame(away_team_tactic_skill_1),pd.DataFrame(away_team_tactic_skill_7)],axis=1),
            num_iteration=model.best_iteration_),decimals=2)
        # On définit une prévision "plancher" à 0.1 (hors cas de forfait)
        xG_dom[(xG_dom<.1) & (home_team_rating_midfield!=1)]=.1
        
        xG_ext=(home_team_rating_midfield==1)*(diff_buts==-5)*5+(home_team_rating_midfield>1)*np.around(model.predict(pd.concat([pd.DataFrame(away_team_rating_midfield**3/(away_team_rating_midfield**3+home_team_rating_midfield**3)),
            pd.DataFrame(.92*(away_team_rating_right_att)**3.5/(away_team_rating_right_att**3.5+(home_team_rating_left_def)**3.5)),
            pd.DataFrame(.92*(away_team_rating_left_att)**3.5/(away_team_rating_left_att**3.5+(home_team_rating_right_def)**3.5)),
            pd.DataFrame(.92*(away_team_rating_mid_att)**3.5/(away_team_rating_mid_att**3.5+(home_team_rating_mid_def)**3.5)),
            pd.DataFrame(.92*away_team_rating_ind_set_pieces_att**3.5/(away_team_rating_ind_set_pieces_att**3.5+home_team_rating_ind_set_pieces_def**3.5)),
            pd.DataFrame(away_team_tactic_skill_1),pd.DataFrame(away_team_tactic_skill_2),pd.DataFrame(away_team_tactic_skill_3),
            pd.DataFrame(away_team_tactic_skill_4),pd.DataFrame(away_team_tactic_skill_7),pd.DataFrame(away_team_tactic_skill_8),
            pd.DataFrame(home_team_tactic_skill_1),pd.DataFrame(home_team_tactic_skill_7)],axis=1),
            num_iteration=model.best_iteration_),decimals=2)
        xG_ext[(xG_ext<.1) & (away_team_rating_midfield!=1)]=.1
        
        
        Liste_matchs=pd.DataFrame(columns=['Home Team','Away Team','Score','xG Home','xG Away','Home win','Draw','Away win','Xpts Home','Xpts Away','Rpts Home','Rpts Away'])
        for i in range(0,nb_matchs):
            Tab_probas=pd.DataFrame(columns=range(0,15))
            Tab_probas.loc[0,0]=norm.cdf(0.5,(xG_dom[i]>=.4)*(xG_dom[i]<2)*(1.1314*xG_dom[i]-.3065)+(xG_dom[i]>=2)*xG_dom[i],
            .0052*xG_dom[i]**3-.0957*xG_dom[i]**2+.5098*xG_dom[i]+.5666)*norm.cdf(0.5,(xG_ext[i]>=.4)*(xG_ext[i]<2)*(1.1314*xG_ext[i]-.3065)+(xG_ext[i]>=2)*xG_ext[i],.0052*xG_ext[i]**3-.0957*xG_ext[i]**2+.5098*xG_ext[i]+.5666)
            for l in range(1,15):
                Tab_probas.loc[0,l]=norm.cdf(0.5,(xG_dom[i]>=.4)*(xG_dom[i]<2)*(1.1314*xG_dom[i]-.3065)+(xG_dom[i]>=2)*xG_dom[i],
                    .0052*xG_dom[i]**3-.0957*xG_dom[i]**2+.5098*xG_dom[i]+.5666)*(norm.cdf(l+0.5,(xG_ext[i]>=.4)*(xG_ext[i]<2)*(1.1314*xG_ext[i]-.3065)+(xG_ext[i]>=2)*xG_ext[i],.0052*xG_ext[i]**3-.0957*xG_ext[i]**2+.5098*xG_ext[i]+.5666)
                    -norm.cdf(l-0.5,(xG_ext[i]>=.4)*(xG_ext[i]<2)*(1.1314*xG_ext[i]-.3065)+(xG_ext[i]>=2)*xG_ext[i],.0052*xG_ext[i]**3-.0957*xG_ext[i]**2+.5098*xG_ext[i]+.5666))
                for k in range(1,15):
                    Tab_probas.loc[k,l]=(norm.cdf(k+0.5,(xG_dom[i]>=.4)*(xG_dom[i]<2)*(1.1314*xG_dom[i]-.3065)+(xG_dom[i]>=2)*xG_dom[i],.0052*xG_dom[i]**3-.0957*xG_dom[i]**2+.5098*xG_dom[i]+.5666)
                    -norm.cdf(k-0.5,(xG_dom[i]>=.4)*(xG_dom[i]<2)*(1.1314*xG_dom[i]-.3065)+(xG_dom[i]>=2)*xG_dom[i],.0052*xG_dom[i]**3-.0957*xG_dom[i]**2+.5098*xG_dom[i]+.5666))*(norm.cdf(l+0.5,(xG_ext[i]>=.4)*(xG_ext[i]<2)*(1.1314*xG_ext[i]-.3065)+(xG_ext[i]>=2)*xG_ext[i],.0052*xG_ext[i]**3-.0957*xG_ext[i]**2+.5098*xG_ext[i]+.5666)
                    -norm.cdf(l-0.5,(xG_ext[i]>=.4)*(xG_ext[i]<2)*(1.1314*xG_ext[i]-.3065)+(xG_ext[i]>=2)*xG_ext[i],.0052*xG_ext[i]**3-.0957*xG_ext[i]**2+.5098*xG_ext[i]+.5666))
            for k in range(1,15):
                Tab_probas.loc[k,0]=(norm.cdf(k+0.5,(xG_dom[i]>=.4)*(xG_dom[i]<2)*(1.1314*xG_dom[i]-.3065)+(xG_dom[i]>=2)*xG_dom[i],.0052*xG_dom[i]**3-.0957*xG_dom[i]**2+.5098*xG_dom[i]+.5666)
                -norm.cdf(k-0.5,(xG_dom[i]>=.4)*(xG_dom[i]<2)*(1.1314*xG_dom[i]-.3065)+(xG_dom[i]>=2)*xG_dom[i],.0052*xG_dom[i]**3-.0957*xG_dom[i]**2+.5098*xG_dom[i]+.5666))*norm.cdf(0.5,(xG_ext[i]>=.4)*(xG_ext[i]<2)*(1.1314*xG_ext[i]-.3065)+(xG_ext[i]>=2)*xG_ext[i],.0052*xG_ext[i]**3-.0957*xG_ext[i]**2+.5098*xG_ext[i]+.5666)
            Proba1=0
            ProbaN=0
            Proba2=0
            for k in range(0,15):
                for l in range(0,15):
                    if k>l:
                        Proba1=Proba1+Tab_probas.loc[k,l]
                    elif k<l:
                        Proba2=Proba2+Tab_probas.loc[k,l]
                    elif k==l:
                        ProbaN=ProbaN+Tab_probas.loc[k,l]
            Liste_matchs.loc[i]=[liste_matchs[i].home_team_name,liste_matchs[i].away_team_name,
                str(liste_matchs[i].home_team_goals)+"-"+str(liste_matchs[i].away_team_goals),xG_dom[i],
                xG_ext[i],str(int(round(Proba1*100,0)))+"%",str(100-(int(round(Proba1*100,0))+int(round(ProbaN*100,0))+int(round(Proba2*100,0)))+int(round(ProbaN*100,0)))+"%",
                str(int(round(Proba2*100,0)))+"%",round(3*Proba1+ProbaN,2),round(3*Proba2+ProbaN,2),
                3*(liste_matchs[i].home_team_goals>liste_matchs[i].away_team_goals)+
                    1*(liste_matchs[i].home_team_goals==liste_matchs[i].away_team_goals),
                3*(liste_matchs[i].away_team_goals>liste_matchs[i].home_team_goals)+
                    1*(liste_matchs[i].home_team_goals==liste_matchs[i].away_team_goals)]
        
        
        classement_dom=pd.DataFrame(Liste_matchs.groupby('Home Team').agg({'Xpts Home':'sum','Rpts Home':'sum'}).reset_index()[['Home Team','Xpts Home','Rpts Home']]).merge(pd.DataFrame(Liste_matchs.groupby('Away Team').agg({'Xpts Away':'sum','Rpts Away':'sum'}).reset_index()[['Away Team','Xpts Away','Rpts Away']]).rename({'Away Team':'Home Team'},axis=1),on='Home Team',how='left').fillna(0)
        classement_ext=pd.DataFrame(Liste_matchs.groupby('Away Team').agg({'Xpts Away':'sum','Rpts Away':'sum'}).reset_index()[['Away Team','Xpts Away','Rpts Away']]).merge(pd.DataFrame(Liste_matchs.groupby('Home Team').agg({'Xpts Home':'sum','Rpts Home':'sum'}).reset_index()[['Home Team','Xpts Home','Rpts Home']]).rename({'Home Team':'Away Team'},axis=1),on='Away Team',how='left').fillna(0)
        classement=pd.concat([classement_dom,classement_ext],axis=0)
        classement['Home Team']=classement['Home Team'].fillna(classement['Away Team'])
        classement['Real points']=classement['Rpts Home']+classement['Rpts Away']
        classement['Expected points']=classement['Xpts Home']+classement['Xpts Away']
        classement['Points difference']=classement['Real points']-classement['Expected points']
        classement=classement.rename({'Home Team':'Team'},axis=1).drop(['Xpts Home','Xpts Away','Rpts Home','Rpts Away','Away Team'],axis=1).sort_values(by='Real points',ascending=0)
        classement=classement.sort_values(by='Expected points',ascending=0).reset_index().drop('index',axis=1).drop_duplicates()
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
        liste_format_HT=re.sub('<style.*?</style>','',Liste_matchs,flags=re.DOTALL)
        liste_format_HT=re.sub('<table.*?>','<table>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<th.*?>','<th>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<td.*?>','<td>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('>\n.*?<','><',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<table>.*?<tr>','<table><tr>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('<tr>.*?<th>','<tr><th>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=re.sub('</th>.*?<th>','</th><th>',liste_format_HT,flags=re.DOTALL)
        liste_format_HT=liste_format_HT.replace('<tbody>','').replace('</tbody>','').replace('</thead>','').replace("<","[").replace(">","]")
        class_format_HT=re.sub('<style.*?</style>','',classement,flags=re.DOTALL)
        class_format_HT=re.sub('<table.*?>','<table>',class_format_HT,flags=re.DOTALL)
        class_format_HT=re.sub('<th.*?>','<th>',class_format_HT,flags=re.DOTALL)
        class_format_HT=re.sub('<td.*?>','<td>',class_format_HT,flags=re.DOTALL)
        class_format_HT=re.sub('>\n.*?<','><',class_format_HT,flags=re.DOTALL)
        class_format_HT=re.sub('<table>.*?<tr>','<table><tr>',class_format_HT,flags=re.DOTALL)
        class_format_HT=re.sub('<tr>.*?<th>','<tr><th>',class_format_HT,flags=re.DOTALL)
        class_format_HT=re.sub('</th>.*?<th>','</th><th>',class_format_HT,flags=re.DOTALL)
        class_format_HT=class_format_HT.replace('<tbody>','').replace('</tbody>','').replace('</thead>','').replace("<","[").replace(">","]")
        return Liste_matchs, classement, liste_format_HT, class_format_HT
    if request.method=='POST':
        Liste_matchs, classement, liste_format_HT, class_format_HT=calcul_pred_league(int(request.form['LeagueID']),int(request.form['Season']))
    else:
        saison=int(request.args.get('Season'))
        ligue=int(request.args.get('LeagueID'))
        Liste_matchs, classement, liste_format_HT, class_format_HT =calcul_pred_league(ligue,saison)
    return render_template('league_pred.html',tables=[classement,Liste_matchs,class_format_HT,liste_format_HT],titles=['SIMULATED RANKING - What is the theorical ranking ?','COMPLETE MATCH LIST - Surprising results are highlighted in yellow','Copy the following code to share the simulated rankings on Hattrick forum :','Copy the following code to share the match list on Hattrick forum :'])
    

# Prédicteur de match customisable
@app.route('/predict_match_cust', methods=("POST", "GET"))
def html_predict_cust():
    def calcul_pred_cust(HomeMidfield, AwayMidfield, HomeRightAtt, HomeLeftAtt, HomeMidAtt, HomeRightDef, HomeLeftDef, HomeMidDef,
        AwayRightAtt, AwayLeftAtt, AwayMidAtt, AwayRightDef, AwayLeftDef, AwayMidDef, HomeIndSPDef, HomeIndSPAtt, AwayIndSPDef,
        AwayIndSPAtt, TacticHome, TacticAway, TacticSkillHome, TacticSkillAway):
        xG_dom=max(0.1,model.predict([[(HomeMidfield*4-3)**3/((HomeMidfield*4-3)**3+(AwayMidfield*4-3)**3),
            .92*(HomeRightAtt*4-3)**3.5/((HomeRightAtt*4-3)**3.5+(AwayLeftDef*4-3)**3.5),
            .92*(HomeLeftAtt*4-3)**3.5/((HomeLeftAtt*4-3)**3.5+(AwayRightDef*4-3)**3.5),
            .92*(HomeMidAtt*4-3)**3.5/((HomeMidAtt*4-3)**3.5+(AwayMidDef*4-3)**3.5),
            .92*(HomeIndSPAtt*4-3)**3.5/((HomeIndSPAtt*4-3)**3.5+(AwayIndSPDef*4-3)**3.5),
            1*(TacticHome=='Pressing')*TacticSkillHome,1*(TacticHome=='Counter-attacks')*TacticSkillHome,
            1*(TacticHome=='Attack in the middle')*TacticSkillHome,1*(TacticHome=='Attack on wings')*TacticSkillHome,
            1*(TacticHome=='Play creatively')*TacticSkillHome,1*(TacticHome=='Long shots')*TacticSkillHome,
            1*(TacticAway=='Pressing')*TacticSkillAway,1*(TacticAway=='Play creatively')*TacticSkillAway]])[0])
        # Extérieur
        xG_ext=max(0.1,model.predict([[(AwayMidfield*4-3)**3/((AwayMidfield*4-3)**3+(HomeMidfield*4-3)**3),
            .92*(AwayRightAtt*4-3)**3.5/((AwayRightAtt*4-3)**3.5+(HomeLeftDef*4-3)**3.5),
            .92*(AwayLeftAtt*4-3)**3.5/((AwayLeftAtt*4-3)**3.5+(HomeRightDef*4-3)**3.5),
            .92*(AwayMidAtt*4-3)**3.5/((AwayMidAtt*4-3)**3.5+(HomeMidDef*4-3)**3.5),
            .92*(AwayIndSPAtt*4-3)**3.5/((AwayIndSPAtt*4-3)**3.5+(HomeIndSPDef*4-3)**3.5),
            1*(TacticAway=='Pressing')*TacticSkillAway, 1*(TacticAway=='Counter-attacks')*TacticSkillAway,
            1*(TacticAway=='Attack in the middle')*TacticSkillAway,1*(TacticAway=='Attack on wings')*TacticSkillAway,
            1*(TacticAway=='Play creatively')*TacticSkillAway,1*(TacticAway=='Long shots')*TacticSkillAway,
            1*(TacticHome=='Pressing')*TacticSkillHome,1*(TacticHome=='Play creatively')*TacticSkillHome]])[0])
        
        # Calcul des probabilités de victoire
        Liste_matchs=pd.DataFrame(columns=['Home Team','Away Team','xG Home','xG Away','Home win','Draw','Away win'])
        Tab_probas=pd.DataFrame(columns=range(0,15))
        Tab_probas.loc[0,0]=norm.cdf(0.5,.0023*xG_dom**3-.0431*xG_dom**2+1.2527*xG_dom-.4389,
            max(0.9,.0032*xG_dom**3-.0662*xG_dom**2+.3828*xG_dom+.7108))*norm.cdf(0.5,.0023*xG_ext**3-.0431*xG_ext**2+1.2527*xG_ext-.4389,max(0.9,.0032*xG_ext**3-.0662*xG_ext**2+.3828*xG_ext+.7108))
        for l in range(1,15):
            Tab_probas.loc[0,l]=norm.cdf(0.5,.0023*xG_dom**3-.0431*xG_dom**2+1.2527*xG_dom-.4389,
                max(0.9,.0032*xG_dom**3-.0662*xG_dom**2+.3828*xG_dom+.7108))*(norm.cdf(l+0.5,.0023*xG_ext**3-.0431*xG_ext**2+1.2527*xG_ext-.4389,max(0.9,.0032*xG_ext**3-.0662*xG_ext**2+.3828*xG_ext+.7108))
                -norm.cdf(l-0.5,.0023*xG_ext**3-.0431*xG_ext**2+1.2527*xG_ext-.4389,max(0.9,.0032*xG_ext**3-.0662*xG_ext**2+.3828*xG_ext+.7108)))
            for k in range(1,15):
                Tab_probas.loc[k,l]=(norm.cdf(k+0.5,.0023*xG_dom**3-.0431*xG_dom**2+1.2527*xG_dom-.4389,max(0.9,.0032*xG_dom**3-.0662*xG_dom**2+.3828*xG_dom+.7108))
                -norm.cdf(k-0.5,.0023*xG_dom**3-.0431*xG_dom**2+1.2527*xG_dom-.4389,max(0.9,.0032*xG_dom**3-.0662*xG_dom**2+.3828*xG_dom+.7108)))*(norm.cdf(l+0.5,.0023*xG_ext**3-.0431*xG_ext**2+1.2527*xG_ext-.4389,max(0.9,.0032*xG_ext**3-.0662*xG_ext**2+.3828*xG_ext+.7108))
                -norm.cdf(l-0.5,.0023*xG_ext**3-.0431*xG_ext**2+1.2527*xG_ext-.4389,max(0.9,.0032*xG_ext**3-.0662*xG_ext**2+.3828*xG_ext+.7108)))
        for k in range(1,15):
            Tab_probas.loc[k,0]=(norm.cdf(k+0.5,.0023*xG_dom**3-.0431*xG_dom**2+1.2527*xG_dom-.4389,max(0.9,.0032*xG_dom**3-.0662*xG_dom**2+.3828*xG_dom+.7108))
                -norm.cdf(k-0.5,.0023*xG_dom**3-.0431*xG_dom**2+1.2527*xG_dom-.4389,max(0.9,.0032*xG_dom**3-.0662*xG_dom**2+.3828*xG_dom+.7108)))*norm.cdf(0.5,.0023*xG_ext**3-.0431*xG_ext**2+1.2527*xG_ext-.4389,max(0.9,.0032*xG_ext**3-.0662*xG_ext**2+.3828*xG_ext+.7108))
        Proba1=0
        ProbaN=0
        Proba2=0
        for k in range(0,15):
            for l in range(0,15):
                if k>l:
                    Proba1=Proba1+Tab_probas.loc[k,l]
                elif k<l:
                    Proba2=Proba2+Tab_probas.loc[k,l]
                elif k==l:
                    ProbaN=ProbaN+Tab_probas.loc[k,l]
        Liste_matchs.loc[0]=['Home Team','Away team',xG_dom,xG_ext,str(int(round(Proba1*100,0)))+"%",str(100-(int(round(Proba1*100,0))+int(round(ProbaN*100,0))+int(round(Proba2*100,0)))+int(round(ProbaN*100,0)))+"%",str(int(round(Proba2*100,0)))+"%"]
    
        # Surprise ou non
        Liste_matchs=Liste_matchs.style.hide_index().set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}]).set_precision(2).render()
        return Liste_matchs
    Liste_matchs=calcul_pred_cust(float(request.form['HomeMidfield']),float(request.form['AwayMidfield']),
        float(request.form['HomeRightAtt']),float(request.form['HomeLeftAtt']),float(request.form['HomeMidAtt']),
        float(request.form['HomeRightDef']),float(request.form['HomeLeftDef']),float(request.form['HomeMidDef']),
        float(request.form['AwayRightAtt']),float(request.form['AwayLeftAtt']),float(request.form['AwayMidAtt']),
        float(request.form['AwayRightDef']),float(request.form['AwayLeftDef']),float(request.form['AwayMidDef']),
        float(request.form['HomeIndSPDef']),float(request.form['HomeIndSPAtt']),float(request.form['AwayIndSPDef']),
        float(request.form['AwayIndSPAtt']),request.form['TacticHome'],request.form['TacticAway'],
        int(request.form['TacticSkillHome']),int(request.form['TacticSkillAway']))
    return render_template('match_pred_cust.html',tables=[Liste_matchs],titles=['MATCH RESULT PROBABILITIES'])

if __name__ == '__main__':
    app.run()
