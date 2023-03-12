from datetime import datetime
import HALoaderDumper as hald
import HAXYProcessing as haxyp
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

def global_eval_NB(y_predict, y_e):
    s = 0
    t = max(1, np.size(y_predict))
    for i in range(0, t):
        if y_e[i] == y_predict[i]:
            s += 1
    print(f"Success rate = {s} / {t}. Id est {round(s / t * 100, 2)}%")
    return s, t

def eval_vec(vec):
    max = 0
    win = 0
    for i in range(0, np.size(vec)):
        if vec[i] > max:
            max = vec[i]
            win = i
    return win


data_directory_path = "/Users/ondrejcikhart/Desktop/Coding-projects/Hockey_analytics/data/"
season_file_path = data_directory_path + "player_matches.json"
data_file_path = data_directory_path + "ppg_data.json"
result_log_path = data_directory_path + "ppg_log.txt"
master_league = 'Tipsport'

dataset_config = {
    'master_league': master_league,
    'season_file': season_file_path,
    'data_file': data_file_path,
    'result_log': result_log_path,
    'range_min': 4,
    'range_max': 100,
    'rand_split': 95
}

### dataset = hald.GenerateData(dataset_config)
dataset = hald.LoadData(dataset_config)

### data_set = {
###     'model' : 'single_player',
###     'x_description' : 'game_played, goals, assists, +-, pt, toi ===== for last season and last 3',
###     'y_description' : 'goals, assists, +- ===== max values 3 and -3',
###     'x_train' : xt_matrix_all,
###     'y_train' : yt_matrix_all,
###     'x_eval' : xe_matrix_all,
###     'y_eval' : ye_matrix_all,
###     'train_log': train_log,
###     'eval_log': eval_log
### }

X_train = np.array(haxyp.Process_X_Season_PPG(dataset['x_train']), dtype='float32')
Y_train = np.array(haxyp.Process_Y_Season_PPG(dataset['y_train']), dtype='int32')
X_eval = np.array(haxyp.Process_X_Season_PPG(dataset['x_eval']), dtype='float32')
Y_eval = np.array(haxyp.Process_Y_Season_PPG(dataset['y_eval']), dtype='int32')

scaler = preprocessing.StandardScaler().fit(X_train)
Xn_train = scaler.transform(X_train)
Xn_eval = scaler.transform(X_eval)

print(f"X_train shape = {X_train.shape}")
print(f"Y_train shape = {Y_train.shape}")

print(f"X_eval shape = {X_eval.shape}")
print(f"Y_eval shape = {Y_eval.shape}")

gnb = GaussianNB()
Y_pred = gnb.fit(Xn_train, Y_train).predict(Xn_eval)

### xt_matrix_all = dataset['x_train']
### for i in range(0, 10):
###     print(f"X{i}: {xt_matrix_all[i]} === {X_train[i]} === Y: {Y_train[i]}")

success_n, total_n = global_eval_NB(Y_pred, Y_eval)

log_f = open(dataset_config['result_log'], 'a')
print(f"===============================================", file = log_f)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file = log_f)
print(f"PPG Data set = {dataset_config['data_file']}\n", file = log_f)
print(f"X_train shape = {X_train.shape}", file = log_f)
print(f"Y_train shape = {Y_train.shape}", file = log_f)
print(f"X_eval shape = {X_eval.shape}", file = log_f)
print(f"Y_eval shape = {Y_eval.shape}", file = log_f)
print(f"Model GaussianNB", file = log_f)
print(f"Success rate = {success_n} / {total_n}. Id est {round(success_n / total_n * 100, 2)}%\n", file = log_f)
