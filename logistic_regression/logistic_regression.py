import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)




class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, max_iterations=100000, l1_strength=0.01, l2_strength=0.01):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        # Hyperparameters
        self.learning_rate = learning_rate # How much to update in each iteration, the step size of gradient descent
        self.max_iterations = max_iterations # Max number of gradient descent iterations
        self.tolerance = 1e-5 # Convergence criterion
        # Use 0.0 for unregularized for l1_strength and or l2_strength
        self.l1_strength = l1_strength # L1 regularization strength (Lasso). Used for punish overfitting 
        self.l2_strength = l2_strength # L2 regularization strength (Ridge). Used for punish overfitting

        # Attributes that will be estimated from the data
        self.weights = np.zeros(1)
        self.bias = 0

        self._last_loss = 0
        self.history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        samples: int = X.shape[0]
        features: int = X.shape[1]

        # Initialize weights
        self.weights = self.random_weights(features)
        self.bias = 0

        for _ in range(0, self.max_iterations):
            # Calculate predictions
            model: np.ndarray = np.dot(X, self.weights.T) + self.bias
            y_pred: np.ndarray = sigmoid(model)

            # Calculate gradients
            dw =  np.dot(X.T, (y_pred - y))  - self.regularization(self.weights)
            db =  np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate loss
            loss = binary_cross_entropy(y, y_pred)
            self.history.append((_, loss)) # TODO: REMOVE
            if self.convergence_criterion(loss):
                break
            self._last_loss = loss
            

    def random_weights(self, shape: int) -> np.ndarray:
        """
        Initializes weights for the classifier
        
        Args:
            shape (int): the number of weights to initialize
        
        Returns:
            A numpy array of shape (shape,)
        """
        return np.random.randn(shape) * 0.01

    def regularization(self, weights: np.ndarray) -> float:
        """
        Calculates the regularization term with L1 and L2 regularization given the weights
        This is used for the gradient descent update and Elastic Net regularization

        Args:
            weights (array<n>): a vector of floats containing the weights
        Returns:
            The regularization term
        """
        return self._l1_regularization(weights) + self._l2_regularization(weights)
    
    def _l1_regularization(self, weights: np.ndarray) -> float:
        """
        Calculates the L1 regularization term given the weights by the Lasso
        This is used for the gradient descent update

    
        Args:
            weights (array<n>): a vector of floats containing the weights
        Returns:
            The regularization term
        """
        return self.l1_strength * np.sign(weights)

    def _l2_regularization(self, weights: np.ndarray) -> float:
        """
        Calculates the L2 regularization term given the weights
        This is used for the gradient descent update
        """
        return  self.l2_strength * np.sum(np.abs(weights))
    
    def change_learning_rate(self, learning_rate: float, loss: float):
        """
        Changes the learning rate based on the loss from the previous epoch
        """
        if loss > self._last_loss:
            self.learning_rate *= 0.90
        else:
            self.learning_rate *= 1.05

    def convergence_criterion(self, loss: float) -> bool:
        """
        Checks if the loss has converged
        
        Args:
            loss (float): the loss from the previous epoch
        
        Returns:
            True if the loss has converged, False otherwise
        """
        return np.abs(self._last_loss - loss) <= self.tolerance 
    
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """

        z: np.ndarray = np.dot(X, self.weights.T) + self.bias
        soft_prediction: np.ndarray = sigmoid(z)
        return soft_prediction
        # hard_prediction = np.where(soft_prediction >= 0.5, 1, 0)
        # return hard_prediction
        # TODO: Implement
        # raise NotImplemented()
        

        
# --- Some utility functions 



def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold=0.5) -> float:
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

# --- Loss functions
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-15) -> float:
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptable
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

import matplotlib.pyplot as plt

def plot_xy_pairs(xy_pairs):
    """
    Plots x,y values from a list of (x,y) value pairs.
    
    Parameters:
    - xy_pairs: List of (x,y) tuples
    """
    
    # Unzip the x and y values
    x_values, y_values = zip(*xy_pairs)
    
    plt.figure(figsize=(10,6))
    plt.scatter(x_values, y_values, color='blue', marker='o', label='Data Points')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Values')
    plt.title('Scatter Plot of X,Y Value Pairs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data_1 = pd.read_csv("logistic_regression\data_1.csv")
    data_1.head()

    # Partition data into independent (feature) and depended (target) variables
    X = data_1[["x0", "x1"]]
    y = data_1["y"]

    # Create and train model.
    model_1 = LogisticRegression()  # <-- Should work with default constructor
    model_1.fit(X, y)

    # Calculate accuracy and cross entropy for (insample) predictions
    y_pred = model_1.predict(X)
    print("Test 1")
    print(f"Accuracy: {binary_accuracy(y_true=y, y_pred=y_pred, threshold=0.5) :.3f}")
    print(f"Cross Entropy: {binary_cross_entropy(y_true=y, y_pred=y_pred) :.3f}")
    plot_xy_pairs(model_1.history)

    # Load second dataset and partition into train/test split
    data_2 = pd.read_csv("logistic_regression\data_2.csv")
    data_2.head()
    data_2_train = data_2.query('split == "train"')
    data_2_test = data_2.query('split == "test"')

    # Partition data into independent (features) and depended (targets) variables
    X_train, y_train = data_2_train[['x0', 'x1']], data_2_train['y']
    X_test, y_test = data_2_test[['x0', 'x1']], data_2_test['y']
    # Fit model (TO TRAIN SET ONLY)
    model_2 = LogisticRegression()  # <--- Feel free to add hyperparameters
    model_2.fit(X_train, y_train)

    # Calculate accuracy and cross entropy for insample predictions 
    y_pred_train = model_2.predict(X_train)
    print('Train')
    print(f'Accuracy: {binary_accuracy(y_true=y_train, y_pred=y_pred_train, threshold=0.5) :.3f}')
    print(f'Cross Entropy:  {binary_cross_entropy(y_true=y_train, y_pred=y_pred_train) :.3f}')

    # Calculate accuracy and cross entropy for out-of-sample predictions
    y_pred_test = model_2.predict(X_test)
    print('\nTest')
    print(f'Accuracy: {binary_accuracy(y_true=y_test, y_pred=y_pred_test, threshold=0.5) :.3f}')
    print(f'Cross Entropy:  {binary_cross_entropy(y_true=y_test, y_pred=y_pred_test) :.3f}')
    plot_xy_pairs(model_2.history)