def Process_X_Season_PPG(xm):
    """
    Transforms xm matrix of original vectors [game_played, goals, assists, +-, pt, toi ===== for last season and last 3]
    to [game_played, points_per_game ===== for last season and last 3]

    Args:
      xm (list)             :X matrix with data to be processed
      
    Returns:
      xm_processed matrix of [game_played, points_per_game ===== for last season and last 3]
    """
    xm_processed = []
    for xi in xm:
        ppg_s = round((xi[1] + xi[2]) / max(1, xi[0]), 2)     #points per game in season
        ppg_lg = round((xi[7] + xi[8]) / max(1, xi[6]), 2)    #points per game in last games
        toipg_lg = round(xi[11] / max(1, xi[6]), 2)  #time on ice per game in last games
        xi_n = [ppg_s, xi[6], ppg_lg, toipg_lg]
        xm_processed.append(xi_n)
    return xm_processed

def Process_Y_Season_PPG(ym):
    """
    Transforms ym matrix of original vectors [goals, assists, +-]
    to vect [points_in_the_game 0, 1, 2+]

    Args:
      ym (list)             :Y matrix with data to be processed
      
    Returns:
      ym_processed (list) as a vector or points in the game
    """
    ym_processed = []
    for yi in ym:
        # ppg = min(2, yi[0] + yi[1])
        ppg = min(2, yi[0] + yi[1])
        ym_processed.append(ppg)
    return ym_processed

def Process_Y_Season_Scored(ym):
    """
    Transforms ym matrix of original vectors [goals, assists, +-]
    to vect [points_in_the_game 0, 1, 2+]

    Args:
      ym (list)             :Y matrix with data to be processed
      
    Returns:
      ym_processed (list) as a vector or points in the game
    """
    ym_processed = []
    for yi in ym:
        ppg = min(1, yi[0] + yi[1])
        ym_processed.append(ppg)
    return ym_processed

def Process_Y_Season_PPG_noMin(ym):
    """
    Transforms ym matrix of original vectors [goals, assists, +-]
    to vect [points_in_the_game 0, 1, ...]

    Args:
      ym (list)             :Y matrix with data to be processed
      
    Returns:
      ym_processed (list) as a vector or points in the game
    """
    ym_processed = []
    for yi in ym:
        # ppg = min(2, yi[0] + yi[1])
        ppg = yi[0] + yi[1]
        ym_processed.append(ppg)
    return ym_processed