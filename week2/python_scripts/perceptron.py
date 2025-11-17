'''
Skeleton code for multiclass perceptron used to solve MNIST problem
'''

import numpy as np
import matplotlib.pyplot as plt


class MutiClassPerceptron:
    '''
    This class implements the basic functionalites of the multi-class perceptron
    '''

    def one_hot_encode(self,y):
        '''
        Function to one-hot-encode the labels
        Inputs:
            - y : N  vector with true labels as integers
        Returns:
            - y_one_hot : N x K matrix with one-hot-encoded labels
        '''
        # Complete your code here
        K = np.max(y) + 1
        y_one_hot = np.eye(K)[y]

        return y_one_hot
    

    def predict(self,w,X):
        '''
        Forward pass of the multi-class perceptron
        Inputs:
            - w : K X D weight matrix
            - X : N x D data matrix
        Returns:
            - y : N x K  vector with predicted labels with one hot encoding
        '''
        # Complete your code here
        scores = self.predict_score(w,X)
        y_pred_int = np.argmax(scores, axis=1)

        y_pred = np.eye(w.shape[0])[y_pred_int]

        return  y_pred
    

    
    def predict_score(self,w,x):
        '''
        Forward pass of the multi-class perceptron that returns the scores
        for each class
        Inputs:
            - w : K x D weight matrix
            - x : N x D data matrix
        Returns:
            - y_score : N x K matrix with predicted scores for each class
        '''

        # Complete your code here
        y_score = x @ w.T
        
        return y_score
    
    def loss(self,y,y_hat):
        '''
        Compute the loss between the true labels and the predicted labels
        Inputs:
            - y : NxK matrix with true labels as 1 or 0
            - y_hat : NxK matrix with predicted labels as 1 or 0

        Returns:
            - loss : int number with the loss
        '''

        # Complete your code here
        y_true = np.argmax(y, axis= 1)
        y_pred = np.argmax(y_hat, axis = 1)

        loss = np.sum(y_true != y_pred)

        return loss

        
    
    def update(self,w,x,y, lr=0.01):
        '''
        Function to obtain the updated weights for the multi-class perceptron
        Inputs:
            - w :  K x D weight matrix
            - x : N x D data matrix
            - y : N x K matrix with  with one hot encoded labels
            - lr : learning rate
        Returns:
            - w : D x K updated weight matrix
        '''
        
        
        y_predict = self.predict(w, x)

        y_int = np.argmax(y, axis=1)
        y_predict_int = np.argmax(y_predict, axis=1)

        for i in range(x.shape[0]):
            if y_int[i] != y_predict_int[i]:
                w[y_int[i]] += lr * x[i]
                w[y_predict_int[i]] -= lr * x[i]

        return w
    
    def train(self,x,y,w_init,n_iter=20):
        '''
        Function to train the multi-class perceptron
        Inputs:
            - x : N x D data matrix
            - y : N  vector with true labels as integers
            - winit : K x D initial weight matrix
            - n_iter : number of iterations
        Returns:
            - w : K x D trained weight matrix
            - loss : float number with the loss
        '''
        
        w = w_init.copy()
        loss_set = []
        # Complete your code here

        for epoch in range(n_iter):
            w = self.update(w, x, y)
            
            y_hat = self.predict(w, x)
            loss = self.loss(y, y_hat)
            loss_set.append(loss)

            


        return w, loss_set

        



if __name__ == '__main__':
    # Load mnist data

    mnist = np.load('./data/mnistData.npz')

    x_train = mnist['arr_0']
    y_train = mnist['arr_1']
    x_test = mnist['arr_2']
    y_test = mnist['arr_3']

    #vectorize the data

    x_train = x_train.reshape(x_train.shape[0],-1)/255

    x_test = x_test.reshape(x_test.shape[0],-1)/255

    #one hot encode the labels

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    print(y_test.shape)

    #initialize the weight matrix
    W_init = np.zeros((y_train.shape[1],x_train.shape[1]))

    #initialize the multi-class perceptron
    model = MutiClassPerceptron()

    y_pred = model.predict(W_init,x_test)

    print(y_pred.shape)

    #train the model
    print('Training the model')
    W,loss = model.train(x_train,y_train,W_init,n_iter=20)

    #predict labels for the test data
    y_pred = model.predict(W,x_test)

    #compute the accuracy
    acc = np.mean(np.argmax(y_pred,axis=1) == np.argmax(y_test,axis=1))
    print('Accuracy: ',acc)

    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.show()


