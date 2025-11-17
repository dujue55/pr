'''
This contains the skeleton code for the random classifier
exercise.
'''

import numpy as np
import matplotlib.pyplot as plt
from perceptron import MutiClassPerceptron




if __name__ == '__main__':

    # Generate random data

    N = 100
    p = 0.5
    d_init = 10

    # Let #X_train = N x d abd #y_train = N x 1 be the training data
    # Using random.randn() generate X_train and random.binomial() generate y_train
    # make sure to one-hot-encode y_train
    # similarly generate X_test and y_test
    # make sure to one-hot-encode y_test
    # Initialize the perceptron with W_int = np.zeros((d,2)) and train it
    # Repeat the above steps for d = np.arange(10,200,10)
    # Plot the accuracy on the training and test set as a function of d

    D_list = np.arange(10, 210, 10)
    n_iter = 20

    train_acc_list = []
    test_acc_list = []

    model = MutiClassPerceptron()

    for D in D_list:
        X_train = np.random.randn(N, D)             
        y_train = np.random.binomial(1, p, N)       

        X_test = np.random.randn(N, D)
        y_test = np.random.binomial(1, p, N)

        y_train_oh = np.eye(2)[y_train]
        y_test_oh = np.eye(2)[y_test]

        W_init = np.zeros((2, D))

        W, loss_set = model.train(X_train, y_train_oh, W_init, n_iter = n_iter)

        y_train_pred = model.predict(W, X_train)
        y_test_pred = model.predict(W, X_test)

        # compute accuracy
        acc_train = np.mean(np.argmax(y_train_pred,axis = 1) == np.argmax(y_train_oh, axis = 1))
        acc_test = np.mean(np.argmax(y_test_pred, axis = 1) == np.argmax(y_test_oh, axis = 1))

        train_acc_list.append(acc_train)
        test_acc_list.append(acc_test)

        print(f"D={D:3d} | Train acc={acc_train:.3f} | Test acc={acc_test:.3f}")
    
    # figure
    plt.figure(figsize=(8, 5))
    plt.plot(D_list, train_acc_list, 'o-', label='Train Accuracy')
    plt.plot(D_list, test_acc_list, 's--', label='Test Accuracy')
    plt.xlabel('Input Dimension D')
    plt.ylabel('Accuracy')
    plt.title('Perceptron Performance on Random Binary Data')
    plt.legend()
    plt.grid(True)
    plt.show()







