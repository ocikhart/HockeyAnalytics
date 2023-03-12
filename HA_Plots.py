import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import HALoaderDumper as hald
import HAXYProcessing as haxyp

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

X_train = np.array(dataset['x_train'], dtype='float32')
Y_train = np.array(dataset['y_train'], dtype='int32')
X_eval = np.array(dataset['x_eval'], dtype='float32')
Y_eval = np.array(dataset['y_eval'], dtype='int32')


plt.style.use('deeplearning.mplstyle')

fig, ax = plt.subplots()
cmap = plt.colormaps["plasma"]
pc = ax.scatter(X_train[:,5], X_train[:,0]+X_train[:,1], c=(Y_train[:,0]+Y_train[:,1]), cmap=cmap)
### pc = ax.scatter(X_train[:,5], X_train[:,1])
### fig.colorbar(pc, ax, extend='both')
fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 7), cmap=cmap),ax=ax, label="Points in Last 3")
ax.set_title('TOI vs Points')
ax.set_xlabel("TOI season")
ax.set_ylabel("Points season")
plt.show()
    
