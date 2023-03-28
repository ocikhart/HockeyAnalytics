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
pip install -r requirements.txt
```

## Usage
Replace `path/to/data.csv` with the path to your CSV data file and run the following command:

```bash
python main.py path/to/data.csv
```

## Example
CSV data example:

```csv
datum,kolo,soutěž,zápas,skóre,G,A,B,+-,TM,TOI,
4.2.2023,45.,CHANCE LIGA,SOK – POR,6:7,1,0,1,-1,4,19:27,
1.2.2023,44.,CHANCE LIGA,POR – LTM,1:3,1,0,1,-2,0,20:16,
...
```


