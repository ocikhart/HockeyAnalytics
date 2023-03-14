import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers as opt

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        ### START CODE HERE ### 
        
        tp = np.sum(np.logical_and(p_val < epsilon, y_val == 1))
        fp = np.sum(np.logical_and(p_val < epsilon, y_val == 0))
        fn = np.sum(np.logical_and(p_val >= epsilon, y_val == 1))
        
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        F1 = 2 * prec * rec / (prec + rec)
        
        ### END CODE HERE ### 
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    
    FWXb = np.matmul(W, X.T).T + b - Y
    CostSquare = np.square(FWXb)
    J = 1/2 * np.sum(R * CostSquare)
    
    WSquare = np.square(W)
    XSquare = np.square(X)
    J += lambda_/2 * np.sum(WSquare)
    J += lambda_/2 * np.sum(XSquare)     
    
    return J

def custom_training_loop(X, W, b, Ynorm, R, lambda_, iterations):
    """
    Runs custom loop collaborative filtering
    Args:
      X (ndarray (num_movies,num_features)) : matrix of item features
      W (ndarray (num_users,num_features))  : matrix of user parameters
      b (ndarray (1, num_users)             : vector of user parameters
      Ynorm (ndarray (num_movies,num_users) : matrix of user ratings of movies, normalized
      R (ndarray (num_movies,num_users)     : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
      iterations (int)
    Returns:
      J (float) : Cost
    """
    optimizer = opt.adam_v2.Adam(learning_rate=1e-1)
    iterations = 200
    lambda_ = 1
    for iter in range(iterations):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost 
        with tf.GradientTape() as tape:

            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient( cost_value, [X,W,b] )

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip(grads, [X,W,b]) )

        # Log periodically.
        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

    return


actual_value = np.array([1, 2, 3])
predicted_value = np.array([1.1, 2.1, 5 ])

# take square of differences and sum them
l2 = np.sum(np.power((actual_value-predicted_value),2))

# take the square root of the sum of squares to obtain the L2 norm
l2_norm = np.sqrt(l2)
print(l2_norm)

predicted_value += actual_value
print(predicted_value)

W = np.array([[1, 0],
              [2, 1],
              [0, 3]])
R = np.array([[1, 0],
              [0, 1],
              [1, 0]])
X = np.array([[5, 3],
              [10, 6]])
b = np.array([[0.5, 0.3, 0.1]])
FWX = np.matmul(W, X.T)
print(FWX)
print(np.square(FWX))
print(np.sum(FWX))
print(FWX*R) 
