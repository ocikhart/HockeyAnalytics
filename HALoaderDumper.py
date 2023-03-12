import os
import codecs, json
import csv
import random

def Save_JSON(json_data, file_path):
    json.dump(json_data, codecs.open(file_path, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=False,
              indent=4) ### this saves the array in .json format
    return

def Load_JSON(file_path):
    # Opening JSON file
    f = open(file_path)
  
    # returns JSON object as a dictionary
    json_data = json.load(f)
    return json_data

def HockeyTimeConvert(hts):
    """
    Convert MM:SS format to float in minutes

    Args:
      hts (string)   :hockey timne string
      
    Returns:
      float
    """
    htf = 0.
    if hts != '' and hts != '-':
        ss = hts.split(':')
        htf = float(ss[0])
        if len(ss) > 1 and ss != '':
            htf += float(ss[1])/60
    htf = round(htf, 2)    
    return htf

def Load_Season_from_PlayerMatchCSV(pm_dir):
    """
    Loads Player x Match data

    Args:
      pm_dir (string)   :player_match directory
      
    Returns:
      dictionary of season (Player x Matches)
    """  
    dir_list = os.listdir(pm_dir)
    players = []
    for file_name in dir_list:
        if not file_name.endswith(".csv"):
            continue
        else:
            p_name = file_name.replace(".csv", "")
        print("Working on file: ", pm_dir + file_name)
        print(p_name)
        with open(pm_dir + file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            pch_season = []
            pelh_season = []
            for row in reader:
                if row['TOI'] == '-':
                    continue
                if row['soutěž'] != 'CHANCE LIGA' and row['soutěž'] != 'Tipsport extraliga':
                    continue
                round = int(row['kolo'].replace(".",""))
                sub_s = str(row['zápas']).split(" ")
                h_team = sub_s[0]
                a_team = sub_s[2]
                sub_s = row['skóre'].split(':')
                h_score = int(sub_s[0])
                a_score = int(sub_s[1])
                pt = HockeyTimeConvert(row['TM'])
                toi = HockeyTimeConvert(row['TOI'])
                m_row = {
                    'round': round,         #round
                    'ht': h_team,           #home team
                    'at': a_team,           #away team
                    'hs': h_score,          #home score
                    'as': a_score,          #away score
                    'g': int(row['G']),     #goals
                    'a': int(row['A']),     #assists
                    '+-': int(row['+-']),   #+-
                    'pt': pt,     #penalty time
                    'toi': toi    #time on ice
                }
                if row['soutěž'] == 'CHANCE LIGA':
                    pch_season.append(m_row)
                else:
                    pelh_season.append(m_row)                    
            ### for per row
                
        player_rec = {
            'name': p_name,
            'Tipsport': pelh_season,
            'CHANCE': pch_season
        }
        players.append(player_rec)
    season_rec = {
        'season': '22-23',
        'players': players
    }    
    return season_rec

def LastGameRec(league, player_rec):
    """
    Find last game of the player in the league

    Args:
      league (string)   :CHANCE or Tipsport
      player_rec (dictionary)
      
    Returns:
      last_round (int), last_match (dictionary)
    """
    l_rec = player_rec[league]
    p = 0
    mlast=[]
    for m in l_rec:
        if m['round'] > p:
            p = m['round']
            mlast = m
    return p, mlast

def MatchInRangeTotal(league, player_rec, range_min=0, range_max=100):
    """
    Calculates total within range of the player in the league

    Args:
      league (string)   :CHANCE or Tipsport
      player_rec (dictionary)
      
    Returns:
      match_in_range_total (list), log_rec (list)
    """
    l_rec = player_rec[league]
    mplayed = 0
    gt = 0
    at = 0
    pmt = 0
    ptt = 0.
    toit = 0.
    log_rec = []
    for m in l_rec:
        if m['round'] >= range_min and m['round'] <range_max:
            mplayed += 1
            gt += m['g']
            at += m['a']
            pmt += m['+-']
            ptt += m['pt']
            toit += m['toi']
            log_rec.append([m['round'], m['g'], m['a'], m['+-']])
    match_ranget = [mplayed, gt, at, pmt, round(ptt, 2), round(toit, 2)]
    return match_ranget, log_rec

def LastButXCouples(league, player_rec, experience, range_min=0, range_max=100, window=1, step=0):
    """
    Calculates total within range of the player in the league

    Args:
      league (string)           :CHANCE or Tipsport
      player_rec (dictionary)   :Player record season / league
      experience (vector)       :gp, gt, at, +-t, ptt, toit
      range_min (int)           :take into account only matches in range
      range_max (int)           :take into account only matches in range
      window (int)              :lenght of window looking back
      step (int)                :skip some matches - by step
      
    Returns:
      [experience, last_but_x] (list) [game_record] (list) [log_records] (list)
    """
    lbx_couples_x = []
    lbx_couples_y = []
    lbx_couples_log = []
    l_rec = player_rec[league]
    skip_pivot = step
    for m in reversed(l_rec):
        if m['round'] < range_min or m['round'] >= range_max:
            continue
        if m['round'] <= window:
            continue
        if skip_pivot == step:
            match_window_total, log_rec = MatchInRangeTotal(league, player_rec, m['round']-window, m['round'])
            x_vec = experience + match_window_total
            g = min(3, m['g'])
            a = min(3, m['a'])
            if m['+-'] > 0:
                plmi = min(3, m['+-'])
            else:
                plmi = max(-3, m['+-'])
            y_vec = [g, a, m['+-']]
            log_vec = [experience, log_rec, [m['round'], g, a, m['+-']]]
            lbx_couples_x.append(x_vec)
            lbx_couples_y.append(y_vec)
            lbx_couples_log.append(log_vec)
            skip_pivot = 0
        else:
            skip_pivot += 1
    return lbx_couples_x, lbx_couples_y, lbx_couples_log

def CreateData_SinglePlayer_Sequence(season_rec, league, range_min = 4, range_pivot = 44, range_max = 100, window = 3, step = 0):
    """
    Creates training and ervaluation data from season records sequentially

    Args:
      season_rec (list)         :list of season of all players
      league (string)           :CHANCE or Tipsport
      range_min (int)           :take into account only matches in range
      range_max (int)           :take into account only matches in range
      range_pivot (int)         :dividing training vs evaluation data
      window (int)              :lenght of window looking back
      step (int)                :skip some matches - by step
      
    Returns:
      {training_set_x, training_set_y, evaluation_set_x, evaluation_set_y} (dictionary)
    """
    season_player_records = season_rec['players']
    train_set_x = []
    train_set_y = []
    train_set_log = []
    eval_set_x = []
    eval_set_y = []
    eval_set_log = []
    for pr in season_player_records:
        ### print(f"Player = {pr['name']}")
        season_total, season_log_rec = MatchInRangeTotal(league, pr)
        train_couples_x, train_couples_y, train_couples_log = LastButXCouples(league, pr, season_total,
                                                                 range_min, range_pivot, window, step)
        train_set_x += train_couples_x
        train_set_y += train_couples_y
        train_set_log += train_couples_log
        eval_couples_x, eval_couples_y, eval_couples_log = LastButXCouples(league, pr, season_total,
                                                                     range_pivot, range_max, window, 0)
        eval_set_x += eval_couples_x
        eval_set_y += eval_couples_y
        eval_set_log += eval_couples_log
    data_set = {
        'model' : 'single_player',
        'x_description' : 'game_played, goals, assists, +-, pt, toi ===== for last season and last 3',
        'y_description' : 'goals, assists, +- ===== max values 3 and -3',
        'x_train' : train_set_x,
        'y_train' : train_set_y,
        'x_eval' : eval_set_x,
        'y_eval' : eval_set_y,
        'train_log': train_set_log,
        'eval_log': eval_set_log
    }
    return data_set

def CreateData_SinglePlayer_RandomSplit(season_rec, league, range_min = 4, rand_pivot = 90, range_max = 100, window = 3, step = 0):
    """
    Creates training and ervaluation data from season records

    Args:
      season_rec (list)         :list of season of all players
      league (string)           :CHANCE or Tipsport
      range_min (int)           :take into account only matches in range
      rand_pivot (int)          :dividing training vs evaluation data by randint(1, 100) by rand_pivot split
      range_max (int)           :take into account only matches in range
      window (int)              :lenght of window looking back
      step (int)                :skip some matches - by step
      
    Returns:
      {training_set_x, training_set_y, evaluation_set_x, evaluation_set_y, train_log, eval_log} (dictionary)
    """
    season_player_records = season_rec['players']
    train_set_x = []
    train_set_y = []
    train_set_log = []
    eval_set_x = []
    eval_set_y = []
    eval_set_log = []
    for pr in season_player_records:
        ### print(f"Player = {pr['name']}")
        season_total, season_log_rec = MatchInRangeTotal(league, pr)
        all_couples_x, all_couples_y, all_couples_log = LastButXCouples(league, pr, season_total,
                                                                 range_min, range_max, window, step)
        for i in range(0, len(all_couples_x)):
            rp = random.randint(1, 100)
            if rp <= rand_pivot:
                train_set_x.append(all_couples_x[i])
                train_set_y.append(all_couples_y[i])
                train_set_log.append(all_couples_log[i])
            else:
                eval_set_x.append(all_couples_x[i])
                eval_set_y.append(all_couples_y[i])
                eval_set_log.append(all_couples_log[i])
    dataset = {
        'model' : 'single_player',
        'x_description' : 'game_played, goals, assists, +-, pt, toi ===== for last season and last 3',
        'y_description' : 'goals, assists, +- ===== max values 3 and -3',
        'x_train' : train_set_x,
        'y_train' : train_set_y,
        'x_eval' : eval_set_x,
        'y_eval' : eval_set_y,
        'train_log': train_set_log,
        'eval_log': eval_set_log
    }
    return dataset

def GenerateData(dataset_config):
    """
    Generates training and evaluation data from season records based on dataset config

    Args:
      dataset_config (dict)             :dataset config record
      
    Returns:
      data_set {training_set_x, training_set_y, evaluation_set_x, evaluation_set_y, train_log, eval_log} (dictionary)
    """
    season = Load_JSON(dataset_config['season_file'])
    dataset = CreateData_SinglePlayer_RandomSplit(season, dataset_config['master_league'],
                                                   dataset_config['range_min'], dataset_config['rand_split'],
                                                   dataset_config['range_max'], dataset_config['lbx_window'])
    Save_JSON(dataset, dataset_config['data_file'])
    return dataset

def LoadData(dataset_config):
    """
    Loads training and evaluation data from season records based on dataset config

    Args:
      dataset_config (dict)             :dataset config record
      
    Returns:
      data_set {training_set_x, training_set_y, evaluation_set_x, evaluation_set_y, train_log, eval_log} (dictionary)
    """
    dataset = Load_JSON(dataset_config['data_file'])
    return dataset