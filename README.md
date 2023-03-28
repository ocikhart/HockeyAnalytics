# Czech Ice Hockey Extraliga Analytics project

This hobby project analyzes player performance data in the Czech Ice Hockey Extraliga for the 2022/2023 season. It reads match-by-match data from CSV files and generates aggregated seasonal data in JSON format. The project aims to predict a player's performance in the next game based on their seasonal performance and performance in the last three games.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Approaches](#approaches)
- [Results](#results)

## Features
- Read and process CSV data files containing individual player performances
- Generate aggregated seasonal JSON data
- Predict player performance using three different approaches: Naive Bayes, Decision Trees, and Neural Networks

## Installation
Clone this repository and install the required dependencies.

```bash
git clone https://github.com/ocikhart/HockeyAnalytics.git
cd HockeyAnalytics
mkdir data
pip install -r requirements.txt
```

## Usage
The models create via `HALoaderDumper.py` several data files like `data/player_matches.json` and `data/ppg_data.json` and running the following command willm execute learning and cross validation:

```bash
python HA_Models_NN.py
```
or

```bash
python HA_Models_NB.py
```

or

```bash
python HA_Trees.py
```


## Example
CSV data example:

```csv
datum,kolo,soutěž,zápas,skóre,G,A,B,+-,TM,TOI,
4.2.2023,45.,CHANCE LIGA,SOK – POR,6:7,1,0,1,-1,4,19:27,
1.2.2023,44.,CHANCE LIGA,POR – LTM,1:3,1,0,1,-2,0,20:16,
...
```

Generated JSON data example:
```json
{
    "season": "22-23",
    "players": [
        {
            "name": "giorgio-estephan",
            "Tipsport": [
                {
                    "round": 50,
                    ...
                }
            ],
            "CHANCE": []
        },
        ...
    ]
}
```

## Approaches
1. Naive Bayes
2. Decision Trees
3. Neural Networks


Neural network model:

```python
model = Sequential(
    [ 
        Dense(24, activation = 'relu', kernel_regularizer=regularizers.L2(model_rec['regularization'])),
        Dense(12, activation = 'relu', kernel_regularizer=regularizers.L2(model_rec['regularization'])),
        Dense(1, activation = 'relu', kernel_regularizer=regularizers.L2(model_rec['regularization']))
    ]
)
```

## Results
The best-performing approach is the neural network, with the following results:

- Success rate = 879 / 1185, i.e., 74.18%
- Precision rate = [0.5374592833876222, 0.0]
- Recall rate = [0.5015197568389058, 0.0]
