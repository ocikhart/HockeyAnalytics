from datetime import datetime
import numpy as np
import HALoaderDumper as hald
import HAXYProcessing as haxyp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def SeekBestDepth(X_train, y_train, X_eval, y_eval, max_depth_list):
    accuracy_list_train = []
    accuracy_list_eval = []
    for max_depth in max_depth_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(max_depth = max_depth, min_samples_split = 200,
                                       random_state = RANDOM_STATE).fit(X_train,y_train) 
        predictions_train = model.predict(X_train) ## The predicted values for the train dataset
        predictions_eval = model.predict(X_eval) ## The predicted values for the test dataset
        accuracy_train = accuracy_score(predictions_train,y_train)
        accuracy_eval = accuracy_score(predictions_eval,y_eval)
        accuracy_list_train.append(accuracy_train)
        accuracy_list_eval.append(accuracy_eval)
    return accuracy_list_train, accuracy_list_eval

def SeekBestMinSamplesSplit(X_train, y_train, X_eval, y_eval, min_samples_split_list):
    accuracy_list_train = []
    accuracy_list_eval = []
    for min_samples_split in min_samples_split_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(max_depth = 32, min_samples_split = min_samples_split,
                                       random_state = RANDOM_STATE).fit(X_train,y_train) 
        predictions_train = model.predict(X_train) ## The predicted values for the train dataset
        predictions_eval = model.predict(X_eval) ## The predicted values for the test dataset
        accuracy_train = accuracy_score(predictions_train,y_train)
        accuracy_eval = accuracy_score(predictions_eval,y_eval)
        accuracy_list_train.append(accuracy_train)
        accuracy_list_eval.append(accuracy_eval)
    return accuracy_list_train, accuracy_list_eval

def Train_vs_Validation(accuracy_list_train, accuracy_list_eval, param_list, x_label):
    plt.title('Train x Validation metrics')
    plt.xlabel(x_label)
    plt.ylabel('accuracy')
    plt.xticks(ticks = range(len(param_list)),labels=param_list) 
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_eval)
    plt.legend(['Train','Validation'])

def global_eval(y_predict, X_e, y_e):
    s = 0
    t_p = [0, 0]
    f_p = [0, 0]
    f_n = [0, 0]
    t = max(1, np.size(y_e))
    for i in range(0, t):
        predict = y_predict[i]
        if y_e[i] == predict:
            s += 1
            if y_e[i] > 0:
                t_p[y_e[i]-1] = t_p[y_e[i]-1] + 1
        else:
            if y_e[i] > 0:
                f_n[y_e[i]-1] = f_n[y_e[i]-1] + 1
                print(f'False negative: X {X_e[i]} ----> Prediction {predict} vs CV {y_e[i]}')
            if predict > 0:
                f_p[predict-1] = f_p[predict-1] + 1
                print(f'False positive: X {X_e[i]} ----> Prediction {predict} vs CV {y_e[i]}')
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


data_directory_path = "/Users/ondrejcikhart/Desktop/Coding-projects/Hockey_analytics/data/"
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

X_train = np.array(dataset['x_train'])
y_train = np.array(haxyp.Process_Y_Season_Scored(dataset['y_train']), dtype='int32')
X_eval = np.array(dataset['x_eval'])
y_eval = np.array(haxyp.Process_Y_Season_Scored(dataset['y_eval']), dtype='int32')

    

RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility

### min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] # If the number is an integer, then it is the actual quantity of samples
### max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.

### accuracy_list_train, accuracy_list_eval = SeekBestDepth(X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
###                                                         max_depth_list=max_depth_list)

### accuracy_list_train, accuracy_list_eval = SeekBestMinSamplesSplit(X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
###                                                                   min_samples_split_list=min_samples_split_list)


### plt.style.use('deeplearning.mplstyle')
### plt.show()

### xgb_model = XGBClassifier(n_estimators = 100, learning_rate = 0.1, verbosity = 1, random_state = RANDOM_STATE)
### xgb_model.fit(X_train, y_train, eval_set = [(X_eval,y_eval)], early_stopping_rounds = 10)
### xgb_model.best_iteration
### y_predict = xgb_model.predict(X_eval)

model = RandomForestClassifier(max_depth = 32, min_samples_split = 200, random_state = RANDOM_STATE).fit(X_train,y_train)
y_predict = model.predict(X_eval)


### print(f"Accuracy score: {accuracy_score(predict_vec, y_eval):.4f}")

global_eval(y_predict=y_predict, X_e=X_eval, y_e=y_eval)
