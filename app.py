from pychpp import CHPP
import pandas as pd
import numpy as np
from scipy.stats import norm
from joblib import load
from flask import Flask, request, render_template
import os
import re
from functools import lru_cache
import logging
from typing import Tuple, List, Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HattrickPredictor:
    """Classe principale pour les prédictions Hattrick"""
    
    def __init__(self):
        self.chpp = self._init_chpp()
        self.model = self._load_model()
        
    def _init_chpp(self) -> CHPP:
        """Initialise la connexion à l'API Hattrick"""
        try:
            return CHPP(
                os.getenv('CONSUMER_KEY'),
                os.getenv('CONSUMER_SECRET'),
                os.getenv('ACCESS_TOKEN_KEY'),
                os.getenv('ACCESS_TOKEN_SECRET')
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de CHPP: {e}")
            raise
    
    def _load_model(self):
        """Charge le modèle de Machine Learning"""
        try:
            return load('lgbm.joblib')
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def _calculate_defensive_pullback(goals_data: tuple) -> Tuple[float, float, float, float]:
        """
        Calcule les facteurs de repli défensif avec mise en cache
        Args:
            goals_data: Tuple des données de buts (immutable pour le cache)
        """
        if not goals_data:
            return 1.0, 1.0, 1.0, 1.0
            
        liste_minutes = [min(goal['minute'], 90) for goal in goals_data]
        liste_hg = [goal['home_goals'] for goal in goals_data]
        liste_ag = [goal['away_goals'] for goal in goals_data]
        
        liste_db = [0] + [hg - ag for hg, ag in zip(liste_hg, liste_ag)]
        liste_db_att = [0.91**(max(db, 1) - 1) for db in liste_db]
        liste_db_def = [1.075**(max(db, 1) - 1) for db in liste_db]
        liste_db_att_ext = [0.91**-min(0, -max(-db, -1) + 1) for db in liste_db]
        liste_db_def_ext = [1.075**-min(0, -max(-db, -1) + 1) for db in liste_db]
        
        liste_md1 = [0] + liste_minutes
        liste_md2 = liste_minutes + [90]
        liste_md = [md2 - md1 for md2, md1 in zip(liste_md2, liste_md1)]
        
        pen_att_dom = sum(att * md for att, md in zip(liste_db_att, liste_md)) / 90
        bon_def_dom = sum(def_val * md for def_val, md in zip(liste_db_def, liste_md)) / 90
        pen_att_ext = sum(att * md for att, md in zip(liste_db_att_ext, liste_md)) / 90
        bon_def_ext = sum(def_val * md for def_val, md in zip(liste_db_def_ext, liste_md)) / 90
        
        return pen_att_dom, bon_def_dom, pen_att_ext, bon_def_ext
    
    def _get_match_data(self, match_id: int):
        """Récupère les données d'un match avec gestion d'erreur"""
        try:
            return self.chpp.match(ht_id=match_id, source="htointegrated")
        except Exception:
            try:
                return self.chpp.match(ht_id=match_id)
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du match {match_id}: {e}")
                raise
    
    def _calculate_xg(self, match_data, is_home: bool, pen_att: float, bon_def: float) -> float:
        """Calcule les xG pour une équipe"""
        diff_buts = match_data.home_team_goals - match_data.away_team_goals
        
        if is_home:
            midfield = max(1, match_data.home_team_rating_midfield)
            opp_midfield = max(1, match_data.away_team_rating_midfield)
            
            # Cas spécial forfait
            if midfield == 1 and diff_buts == 5:
                return 5.0
            if midfield == 1:
                return 0.1
                
            features = [
                midfield**3 / (midfield**3 + opp_midfield**3),
                0.92 * (match_data.home_team_rating_right_att / pen_att)**3.5 / 
                ((match_data.home_team_rating_right_att / pen_att)**3.5 + 
                 (match_data.away_team_rating_left_def / bon_def)**3.5),
                0.92 * (match_data.home_team_rating_left_att / pen_att)**3.5 / 
                ((match_data.home_team_rating_left_att / pen_att)**3.5 + 
                 (match_data.away_team_rating_right_def / bon_def)**3.5),
                0.92 * (match_data.home_team_rating_mid_att / pen_att)**3.5 / 
                ((match_data.home_team_rating_mid_att / pen_att)**3.5 + 
                 (match_data.away_team_rating_mid_def / bon_def)**3.5),
                0.92 * match_data.home_team_rating_ind_set_pieces_att**3.5 / 
                (match_data.home_team_rating_ind_set_pieces_att**3.5 + 
                 match_data.away_team_rating_ind_set_pieces_def**3.5),
                1 * (match_data.home_team_tactic_type == '1') * match_data.home_team_tactic_skill,
                1 * (match_data.home_team_tactic_type == '2') * match_data.home_team_tactic_skill,
                1 * (match_data.home_team_tactic_type == '3') * match_data.home_team_tactic_skill,
                1 * (match_data.home_team_tactic_type == '4') * match_data.home_team_tactic_skill,
                1 * (match_data.home_team_tactic_type == '7') * match_data.home_team_tactic_skill,
                1 * (match_data.home_team_tactic_type == '8') * match_data.home_team_tactic_skill,
                1 * (match_data.away_team_tactic_type == '1') * match_data.away_team_tactic_skill,
                1 * (match_data.away_team_tactic_type == '7') * match_data.away_team_tactic_skill
            ]
        else:
            midfield = max(1, match_data.away_team_rating_midfield)
            opp_midfield = max(1, match_data.home_team_rating_midfield)
            
            # Cas spécial forfait
            if midfield == 1 and diff_buts == -5:
                return 5.0
            if midfield == 1:
                return 0.1
                
            features = [
                midfield**3 / (midfield**3 + opp_midfield**3),
                0.92 * (match_data.away_team_rating_right_att / pen_att)**3.5 / 
                ((match_data.away_team_rating_right_att / pen_att)**3.5 + 
                 (match_data.home_team_rating_left_def / bon_def)**3.5),
                0.92 * (match_data.away_team_rating_left_att / pen_att)**3.5 / 
                ((match_data.away_team_rating_left_att / pen_att)**3.5 + 
                 (match_data.home_team_rating_right_def / bon_def)**3.5),
                0.92 * (match_data.away_team_rating_mid_att / pen_att)**3.5 / 
                ((match_data.away_team_rating_mid_att / pen_att)**3.5 + 
                 (match_data.home_team_rating_mid_def / bon_def)**3.5),
                0.92 * match_data.away_team_rating_ind_set_pieces_att**3.5 / 
                (match_data.away_team_rating_ind_set_pieces_att**3.5 + 
                 match_data.home_team_rating_ind_set_pieces_def**3.5),
                1 * (match_data.away_team_tactic_type == '1') * match_data.away_team_tactic_skill,
                1 * (match_data.away_team_tactic_type == '2') * match_data.away_team_tactic_skill,
                1 * (match_data.away_team_tactic_type == '3') * match_data.away_team_tactic_skill,
                1 * (match_data.away_team_tactic_type == '4') * match_data.away_team_tactic_skill,
                1 * (match_data.away_team_tactic_type == '7') * match_data.away_team_tactic_skill,
                1 * (match_data.away_team_tactic_type == '8') * match_data.away_team_tactic_skill,
                1 * (match_data.home_team_tactic_type == '1') * match_data.home_team_tactic_skill,
                1 * (match_data.home_team_tactic_type == '7') * match_data.home_team_tactic_skill
            ]
        
        return max(0.1, self.model.predict([features])[0])
    
    @staticmethod
    def _calculate_probabilities(xg_home: float, xg_away: float) -> Tuple[float, float, float]:
        """Calcule les probabilités de victoire, nul, défaite"""
        def get_params(xg):
            if 0.4 <= xg < 2:
                mean = 1.1314 * xg - 0.3065
            else:
                mean = xg
            std = 0.0052 * xg**3 - 0.0957 * xg**2 + 0.5098 * xg + 0.5666
            return mean, std
        
        mean_home, std_home = get_params(xg_home)
        mean_away, std_away = get_params(xg_away)
        
        proba_home_win = 0
        proba_draw = 0
        proba_away_win = 0
        
        for k in range(15):
            for l in range(15):
                if k == 0:
                    prob_k = norm.cdf(0.5, mean_home, std_home)
                else:
                    prob_k = norm.cdf(k + 0.5, mean_home, std_home) - norm.cdf(k - 0.5, mean_home, std_home)
                
                if l == 0:
                    prob_l = norm.cdf(0.5, mean_away, std_away)
                else:
                    prob_l = norm.cdf(l + 0.5, mean_away, std_away) - norm.cdf(l - 0.5, mean_away, std_away)
                
                prob_score = prob_k * prob_l
                
                if k > l:
                    proba_home_win += prob_score
                elif k < l:
                    proba_away_win += prob_score
                else:
                    proba_draw += prob_score
        
        return proba_home_win, proba_draw, proba_away_win
    
    def predict_match(self, match_id: int) -> pd.DataFrame:
        """Prédiction d'un match individuel"""
        try:
            match_data = self._get_match_data(match_id)
            
            # Calcul du repli défensif avec cache
            goals_tuple = tuple(match_data.goals) if match_data.goals else tuple()
            pen_att_dom, bon_def_dom, pen_att_ext, bon_def_ext = self._calculate_defensive_pullback(goals_tuple)
            
            # Calcul des xG
            xg_home = self._calculate_xg(match_data, True, pen_att_dom, bon_def_ext)
            xg_away = self._calculate_xg(match_data, False, pen_att_ext, bon_def_dom)
            
            # Calcul des probabilités
            prob_home, prob_draw, prob_away = self._calculate_probabilities(xg_home, xg_away)
            
            # Score final
            try:
                final_score = f"{match_data.goals[-1]['home_goals']}-{match_data.goals[-1]['away_goals']}"
                home_goals = match_data.goals[-1]['home_goals']
                away_goals = match_data.goals[-1]['away_goals']
            except (IndexError, KeyError):
                final_score = "0-0"
                home_goals = away_goals = 0
            
            # Détection des surprises
            is_surprise = (
                (home_goals >= away_goals and prob_away >= 0.5) or
                (home_goals <= away_goals and prob_home >= 0.5)
            )
            
            result_df = pd.DataFrame({
                'Home Team': [match_data.home_team_name],
                'Away Team': [match_data.away_team_name],
                'Score': [final_score],
                'xG Home': [xg_home],
                'xG Away': [xg_away],
                'Home win': [f"{int(round(prob_home * 100))}%"],
                'Draw': [f"{int(round(prob_draw * 100))}%"],
                'Away win': [f"{int(round(prob_away * 100))}%"],
                'Surprise': [1 if is_surprise else 0]
            })
            
            return result_df
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction du match {match_id}: {e}")
            raise

# Fonctions utilitaires
def highlight_surprises(s, column):
    """Fonction pour surligner les surprises dans le DataFrame"""
    is_surp = pd.Series(data=False, index=s.index)
    is_surp[column] = s.loc[column] == 1
    return ['font-weight:bold; background-color:yellow; color:black' if is_surp.any() else '' for v in is_surp]

def format_table_for_hattrick(styled_df: str) -> str:
    """Formate un tableau stylé pour le forum Hattrick"""
    # Suppression du CSS
    formatted = re.sub(r'<style.*?</style>', '', styled_df, flags=re.DOTALL)
    # Simplification des balises
    formatted = re.sub(r'<table.*?>', '<table>', formatted, flags=re.DOTALL)
    formatted = re.sub(r'<th.*?>', '<th>', formatted, flags=re.DOTALL)
    formatted = re.sub(r'<td.*?>', '<td>', formatted, flags=re.DOTALL)
    # Nettoyage des espaces
    formatted = re.sub(r'>\s*<', '><', formatted, flags=re.DOTALL)
    # Simplification de la structure
    formatted = re.sub(r'<table>.*?<tr>', '<table><tr>', formatted, flags=re.DOTALL)
    formatted = re.sub(r'<tr>.*?<th>', '<tr><th>', formatted, flags=re.DOTALL)
    formatted = re.sub(r'</th>.*?<th>', '</th><th>', formatted, flags=re.DOTALL)
    # Suppression des balises body
    formatted = formatted.replace('<tbody>', '').replace('</tbody>', '')
    formatted = formatted.replace('</thead>', '')
    # Conversion pour forum
    formatted = formatted.replace('<', '[').replace('>', ']')
    
    return formatted

# Application Flask
app = Flask(__name__)

# Instance globale du prédicteur (initialisée une seule fois)
predictor = None

def get_predictor():
    """Récupère l'instance du prédicteur (pattern singleton)"""
    global predictor
    if predictor is None:
        predictor = HattrickPredictor()
    return predictor

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Erreur serveur: {e}")
    return render_template('index_err.html'), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('index_err.html'), 404

@app.route('/', methods=['GET', 'POST'])
def html_table():
    return render_template('index.html')

@app.route('/predict_match', methods=['GET', 'POST'])
def html_predict():
    try:
        if request.method == 'POST':
            match_id = int(request.form['MatchID'])
        else:
            match_id = int(request.args.get('MatchID'))
        
        pred = get_predictor()
        result_df = pred.predict_match(match_id)
        
        # Style du DataFrame
        styled_df = result_df.style.hide_index().hide_columns(['Surprise']).set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ]).apply(highlight_surprises, column=['Surprise'], axis=1).set_precision(2)
        
        # Rendu HTML
        html_table = styled_df.render()
        forum_format = format_table_for_hattrick(html_table)
        
        return render_template('match_pred.html', 
                             tables=[html_table, forum_format],
                             titles=['MATCH RESULT PROBABILITIES', 
                                   'Copy the following code to share the result on Hattrick forum :'])
    
    except Exception as e:
        logger.error(f"Erreur dans predict_match: {e}")
        return render_template('index_err.html'), 500

@app.route('/predict_match_cust', methods=['POST'])
def html_predict_cust():
    try:
        # Récupération des paramètres du formulaire
        params = {
            'home_midfield': float(request.form['HomeMidfield']),
            'away_midfield': float(request.form['AwayMidfield']),
            'home_right_att': float(request.form['HomeRightAtt']),
            'home_left_att': float(request.form['HomeLeftAtt']),
            'home_mid_att': float(request.form['HomeMidAtt']),
            'home_right_def': float(request.form['HomeRightDef']),
            'home_left_def': float(request.form['HomeLeftDef']),
            'home_mid_def': float(request.form['HomeMidDef']),
            'away_right_att': float(request.form['AwayRightAtt']),
            'away_left_att': float(request.form['AwayLeftAtt']),
            'away_mid_att': float(request.form['AwayMidAtt']),
            'away_right_def': float(request.form['AwayRightDef']),
            'away_left_def': float(request.form['AwayLeftDef']),
            'away_mid_def': float(request.form['AwayMidDef']),
            'home_sp_def': float(request.form['HomeIndSPDef']),
            'home_sp_att': float(request.form['HomeIndSPAtt']),
            'away_sp_def': float(request.form['AwayIndSPDef']),
            'away_sp_att': float(request.form['AwayIndSPAtt']),
            'tactic_home': request.form['TacticHome'],
            'tactic_away': request.form['TacticAway'],
            'tactic_skill_home': int(request.form['TacticSkillHome']),
            'tactic_skill_away': int(request.form['TacticSkillAway'])
        }
        
        pred = get_predictor()
        result_df = pred.predict_custom_match(**params)
        
        styled_df = result_df.style.hide_index().set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ]).set_precision(2)
        
        return render_template('match_pred_cust.html', 
                             tables=[styled_df.render()],
                             titles=['MATCH RESULT PROBABILITIES'])
    
    except Exception as e:
        logger.error(f"Erreur dans predict_match_cust: {e}")
        return render_template('index_err.html'), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
