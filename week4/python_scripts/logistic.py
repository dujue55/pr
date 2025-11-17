"""
This script contains the functions for logistic regression assignment
"""

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    The class which contains the functions for logistic regression
    """


    @staticmethod
    def sigmoid(z: np.array) -> np.array:
        """
        Calculates the sigmoid of the input
        Inputs:
            - z : vector of size N
        Outputs:
            - y : vector of size N containing the sigmoid of x
        """
        # Complete your code here
        y = 1 / (1 + np.exp(-z)) # input 是一个向量，由于broadcast机制，自动扩展

        return y
    
    @staticmethod
    def predict_score(X: np.array, w: np.array) -> np.array:
        """
        predicts the score for the data
        Inputs: 
            - X : N x D data matrix
            - w : D size weight vector
        Outputs:
            - y_pred : Vector of size N containing the predicted scores
        """

        # Complete your code here
        z = X @ w # shape: (N,)
        y_pred = LogisticRegression.sigmoid(z)
        return y_pred
    

    @staticmethod
    def predict(X: np.array, w: np.array, threshold: float = 0.5) -> np.array:
        """
        predicts the labels for the data
        Inputs: 
            - X : N x D data matrix
            - w : D size weight vector
            - threshold : threshold for the sigmoid function
        Outputs:
            - y_pred : Vector of size N containing the predicted labels as 1 or 0
        """

        # Complete your code here
        
        score = LogisticRegression.predict_score(X, w)
        y_pred = (score >= threshold).astype(int) #.astype是把一个布尔数组变成int数组

        # y_pred = np.zeros_like(score)
        # y_pred[score >= threshold] = 1

        return y_pred
    

    @staticmethod
    def cross_entropy_loss(y_pred_score: np.array, y_true: np.array) -> float:
        """
        Calculates the cross entropy loss for the data
        Inputs:
            - y_pred_score : vector of size N containing the predicted scores
            - y_true : vector of size N containing the true labels as 1 or 0
        Outputs:
            - loss : cross entropy loss
        """

        # Complete your code here
        eps = 1e-12
        y_pred_score = np.clip(y_pred_score, eps, 1 - eps)

        # 逐元素计算
        loss_terms = y_true * np.log(y_pred_score) + (1 - y_true) * np.log(1 - y_pred_score)

        # 求平均并取负号
        loss = -np.mean(loss_terms)

        return loss
    
    @staticmethod
    def gradient(X: np.array, w: np.array, y_true: np.array) -> np.array:
        """
        Calculates the gradient of the loss function
        Inputs:
            - X : N x D data matrix
            - w : D size weight vector
            - y_true : vector of size N containing the true labels as 1 or 0
        Outputs:
            - grad : D size vector containing the gradient of the loss function
        """

        # Complete your code here
        y_pred = LogisticRegression.predict_score(X, w)
        N = X.shape[0]
        grad = (X.T @ (y_pred - y_true)) / N ##需要推导公式
        return grad
    
    @staticmethod
    def train(X: np.array, w_init: np.array,y_true: np.array,
               epochs: int, lr: float) -> Tuple[np.array, List[float]]:
        """
        Use gradient descent to train the logistic regression model
        Inputs:
            - X : N x D data matrix
            - w_init : D size initial weight vector
            - y_true : vector of size N containing the true labels as 1 or 0
            - epochs : number of epochs to train
            - lr : learning rate
        Outputs:
            - w : D size weight vector
            - losses : list containing the loss at each epoch 
        """

        #Complete your code here
        w = w_init.copy()
        losses = []

        # Gradient Descent loop
        for i in range(epochs):
            # 1. forward pass
            y_pred = LogisticRegression.predict_score(X, w)

            # 2. compute loss
            loss = LogisticRegression.cross_entropy_loss(y_pred, y_true)
            losses.append(loss)

            # 3. compute gradient
            grad = LogisticRegression.gradient(X, w, y_true)

            # 4. update weights
            w = w - lr * grad

        return w, losses
    


if __name__ == "__main__":
    # Test the logistic regression class on a toy dataset
    # Create a toy dataset
    np.random.seed(0) # Setting the seed to get consistent results
    N = 100
    D = 2
    X_data = np.random.randn(N, D)
    w_true = np.random.randn(D)
    y_data = np.where(X_data @ w_true > 0, 1, 0)


    # Check outputs with the toy dataset
    y_est = LogisticRegression.predict(X_data, w_true)

    if np.all(y_est == y_data):
        print("Passed the predict function test")

    # Testing Cross Entropy

    y1 = np.array([0.5,0.5])
    y2 = np.array([1,0])

    loss = LogisticRegression.cross_entropy_loss(y1, y2)

    if np.isclose(loss, 0.6931471805599453):
        print("Passed the cross entropy test")


    # Testing the gradient function
    w_init = np.zeros(D)
    grad = LogisticRegression.gradient(X_data, w_init, y_data)

    # Note that this for the case when the seed is set to 0
    if np.all(np.isclose(grad, np.array([0.35061729,0.2135946 ]))):
        print("Passed the gradient test")

    # Testing the train function

    w_init = np.zeros(D)


    X_train = np.random.randn(N, D)
    y_train = np.where(X_train @ w_true > 0, 1, 0)


    X_test = np.random.randn(N, D)
    y_test = np.where(X_test @ w_true > 0, 1, 0)



    y_pred_init = LogisticRegression.predict(X_test, w_init)
    accuracy_init = np.mean(y_pred_init == y_test)


    w_train, losses = LogisticRegression.train(X_train, w_init,
                                               y_train, epochs=100, lr=0.1)
    y_test_predict = LogisticRegression.predict(X_test, w_train)
    accuracy_final = np.mean(y_test_predict == y_test)

    print("Initial accuracy is ", accuracy_init)
    print("Final accuracy is ", accuracy_final)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    



    