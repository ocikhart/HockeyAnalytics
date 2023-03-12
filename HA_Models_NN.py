from datetime import datetime
import HALoaderDumper as hald
import HAXYProcessing as haxyp
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizers import adam_v2 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import regularizers

def global_eval_logistic(Y_p, Y_e):
    s = 0
    t = max(1, np.size(Y_p))
    for i in range(0, t):
        if Y_e[i] == 1 and Y_p[i] >= 0.5:
            s += 1
        if Y_e[i] == 0 and Y_p[i] < 0.5:
            s += 1
    print(f"Success rate = {s} / {t}. Id est {round(s / t * 100, 2)}%")
    return s, t

def global_eval_regression(Y_p, X_e, Y_e, max_val = 2, trsh = 0.5):
    s = 0
    t_p = [0, 0]
    f_p = [0, 0]
    f_n = [0, 0]
    t = max(1, np.size(Y_e))
    for i in range(0, t):
        predict = round(min(max_val, eval_single_val(float(Y_p[i]), trsh)))
        if Y_e[i] == predict:
            s += 1
            if Y_e[i] > 0:
                t_p[Y_e[i]-1] = t_p[Y_e[i]-1] + 1
        else:
            if Y_e[i] > 0:
                f_n[Y_e[i]-1] = f_n[Y_e[i]-1] + 1
                print(f'False negative: X {X_e[i]} ----> Prediction {predict} vs CV {Y_e[i]}')
            if predict > 0:
                f_p[predict-1] = f_p[predict-1] + 1
                print(f'False positive: X {X_e[i]} ----> Prediction {predict} vs CV {Y_e[i]}')
    precision = [t_p[0]/max(1, (t_p[0] + f_p[0])), t_p[1]/max(1, (t_p[1] + f_p[1]))]
    recall = [t_p[0]/max(1, (t_p[0] + f_n[0])), t_p[1]/max(1, (t_p[1] + f_n[1]))]
    eval = {
        'success': s,
        'total': t,
        'precision': precision,
        'recall': recall
    }        
    print(f"Success rate = {s} / {t}. Id est {round(s / t * 100, 2)}%")
    print(f"Precision rate = {precision}%")
    print(f"Recall rate = {recall}%")
    return eval


def eval_vec(vec):
    max = 0
    win = 0
    for i in range(0, np.size(vec)):
        if vec[i] > max:
            max = vec[i]
            win = i
    return win

def eval_single_val(val, trsh = 0.5):
    seed_val = round(val)
    if val - seed_val > trsh:
        seed_val += 1
    elif val - seed_val <= trsh - 1:
        seed_val -= 1
    return seed_val

def global_eval_softmax(y_predict, y_e):
    s = 0
    t_p = [0, 0]
    f_p = [0, 0]
    f_n = [0, 0]
    t = max(1, np.size(y_e))
    for i in range(0, t):
        predict = eval_vec(y_predict[i])
        if y_e[i] == predict:
            s += 1
            if y_e[i] > 0:
                t_p[y_e[i]-1] = t_p[y_e[i]-1] + 1
        else:
            if y_e[i] > 0:
                f_n[y_e[i]-1] = f_n[y_e[i]-1] + 1
                ### print(f'False negative: Prediction {predict} vs CV {y_e[i]}')
            if predict > 0:
                f_p[predict-1] = f_p[predict-1] + 1
                ### print(f'False positive: Prediction {predict} vs CV {y_e[i]}')
    precision = [t_p[0]/max(1, (t_p[0] + f_p[0])), t_p[1]/max(1, (t_p[1] + f_p[1]))]
    recall = [t_p[0]/max(1, (t_p[0] + f_n[0])), t_p[1]/max(1, (t_p[1] + f_n[1]))]
    eval = {
        'success': s,
        'total': t,
        'precision': precision,
        'recall': recall
    }        
    print(f"Success rate = {s} / {t}. Id est {round(s / t * 100, 2)}%")
    print(f"Precision rate = {precision}%")
    print(f"Recall rate = {recall}%")
    return eval


data_directory_path = "/Users/ondrejcikhart/Desktop/Projects/HockeyAnalytics/data/"
season_file_path = data_directory_path + "player_matches.json"
data_file_path = data_directory_path + "ppg_data_linear.json"
result_log_path = data_directory_path + "ppg_log.txt"
master_league = 'Tipsport'

dataset_config = {
    'master_league': master_league,
    'season_file': season_file_path,
    'data_file': data_file_path,
    'result_log': result_log_path,
    'range_min': 4,
    'range_max': 100,
    'lbx_window': 3,
    'rand_split': 90
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

### X_train = np.array(Process_X_Season_PPG(data_set['x_train']), dtype='float32')
### Y_train = np.array(Process_Y_Season_PPG(data_set['y_train']), dtype='int32')
### X_eval = np.array(Process_X_Season_PPG(data_set['x_eval']), dtype='float32')
### Y_eval = np.array(Process_Y_Season_PPG(data_set['y_eval']), dtype='int32')

### X_train = np.array(haxyp.Process_X_Season_PPG(dataset['x_train']), dtype='float32')
### Y_train = np.array(haxyp.Process_Y_Season_PPG_noMin(dataset['y_train']), dtype='int32')
### X_eval = np.array(haxyp.Process_X_Season_PPG(dataset['x_eval']), dtype='float32')
### Y_eval = np.array(haxyp.Process_Y_Season_PPG(dataset['y_eval']), dtype='int32')

X_train = np.array(haxyp.Process_X_Season_PPG(dataset['x_train']), dtype='float32')
Y_train = np.array(haxyp.Process_Y_Season_PPG_noMin(dataset['y_train']), dtype='int32')
X_eval = np.array(haxyp.Process_X_Season_PPG(dataset['x_eval']), dtype='float32')
Y_eval = np.array(haxyp.Process_Y_Season_Scored(dataset['y_eval']), dtype='int32')

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X_train)  # learns mean, variance
Xn_train = norm_l(X_train)
Xn_eval = norm_l(X_eval)

print(f"X_train shape = {Xn_train.shape}")
print(f"Y_train shape = {Y_train.shape}")

print(f"X_eval shape = {Xn_eval.shape}")
print(f"Y_eval shape = {Y_eval.shape}")

model_rec ={
    'regularization': 0.001,
    'adam': 0.01,
    'epochs': 20,
}

model = Sequential(
    [ 
        Dense(24, activation = 'relu', kernel_regularizer=regularizers.L2(model_rec['regularization'])),
        Dense(12, activation = 'relu', kernel_regularizer=regularizers.L2(model_rec['regularization'])),
        ### Dense(2, activation = 'relu'),
        ### Dense(1, activation = 'sigmoid')    # < softmax activation here
        Dense(1, activation = 'relu', kernel_regularizer=regularizers.L2(model_rec['regularization']))    # < softmax activation here
    ]
)

model.compile(
    ### loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- softmax
    ### loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  #<-- sigmoid
    ### loss=tf.keras.losses.SparseCategoricalCrossentropy(),   #<-- softmax
    loss=tf.keras.losses.MeanSquaredError(),
    ### loss=tf.keras.losses.BinaryCrossentropy(),              #<-- sigmoid
    optimizer=adam_v2.Adam(model_rec['adam']),
)

model.fit(
    Xn_train, Y_train,
    epochs = model_rec['epochs']
)

Y_predict = model.predict(Xn_eval)

### xt_matrix_all = dataset['x_train']
### for i in range(0, 10):
###     print(f"X{i}: {xt_matrix_all[i]} === {X_train[i]} === Y: {Y_train[i]}")

threshold = 0.6
eval_rates = global_eval_regression(Y_predict, X_eval, Y_eval, max_val=1, trsh=threshold)
### global_eval_logistic(predict, Y_eval)

log_f = open(dataset_config['result_log'], 'a')
print(f"===============================================", file = log_f)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file = log_f)
print(f"Scored Data set = {dataset_config['data_file']}\n", file = log_f)
print(f"X_train shape = {X_train.shape}", file = log_f)
print(f"Y_train shape = {Y_train.shape}", file = log_f)
print(f"X_eval shape = {X_eval.shape}", file = log_f)
print(f"Y_eval shape = {Y_eval.shape}", file = log_f)
print(f"Regularization L2", file = log_f)
print(f"Model 24 x relu ---> 12 x relu ---> 1 x relu", file = log_f)
print(f"Params = {model_rec}", file = log_f)
print(f"Evaluation threshold = {threshold}", file = log_f)
print(f"Success rate = {eval_rates['success']} / {eval_rates['total']}. Id est {round(eval_rates['success'] / eval_rates['total'] * 100, 2)}%", file = log_f)
print(f"Precision rate = {eval_rates['precision']}", file = log_f)
print(f"Recall rate = {eval_rates['recall']}\n", file = log_f)
