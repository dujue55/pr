'''This script contains the skeleton code for the sample assignment. 
You need to complete the functions below, you are required to complete the lines where it
says "Complete your code here". You are not allowed to use 
any other libraries other than the ones mentioned below.'''
from typing import Tuple
import numpy as np


class Sample:
    """Class containing the functions to be implemented for the sample assignment"""

    def check_span(self, a_matrix: np.ndarray, y_vector: np.ndarray) -> bool:
        """Check Whether the y is present in the span of columns of A, 
        return true if this is the case, else false"""

        # Complete your code here
        rank_A = np.linalg.matrix_rank(a_matrix)
        rank_Ay = np.linalg.matrix_rank(np.column_stack((a_matrix, y_vector))) #np.matrix_stack生成增广矩阵，参数是一个
        return rank_A == rank_Ay

    def estimate_mean_and_covariance(self, x_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x_matrix is an array of size nxk, with n samples of dimension k
        Retun mean followed by covariance"""
        n = x_matrix.shape[0]
        
        # Complete your code here
        mean = np.mean(x_matrix, axis =0)
        certened_matrix = x_matrix - mean # broadcasting机制会自动扩展维度
        covariance = np.cov(certened_matrix, rowvar= False) # np.cov是计算协方差矩阵的方法，默认输入的矩阵是k*n（rowvar = True）,这里输入的矩阵维度不匹配，要设置rowvar = False
        return mean, covariance





if __name__ == '__main__':
    # To check the correctness of the implementations
    # Note this does only a few checks and is not exhaustive
    # You may want to add more tests to check correctness of your implementations
    # However, the autograder will use more tests than the ones provided here
    EPS = 1e-10

    sampleObject = Sample()

    # Checking the span function
    A = np.eye(3)
    y = np.random.randn(3)
    if sampleObject.check_span(A, y):
        print('checkSpan Implementation OK')
    else:
        print('checkSpan Implementation incorrect')

    # Testing mean and covariance on a sample data
    X = np.array([[4, 2, 0.6], [4.2, 2.1, 0.59], [3.9, 2.0, 0.58], [
                 4.3, 2.1, 0.62], [4.1, 2.2, 0.63]])

    true_mean = np.array([4.1, 2.08, 0.604])
    true_covariance = np.array(
        [[0.025, 0.0075, 0.00175], [0.0075, 0.007, 0.00135], [0.00175, 0.00135, 0.00043]])
    estimate_mean, estimate_covariance = sampleObject.estimate_mean_and_covariance(
        X)

    if np.linalg.norm(estimate_mean-true_mean) < EPS:
        print('Mean Implementation OK')
    else:
        print('Mean Implementation incorrect')
    if np.linalg.norm(true_covariance-estimate_covariance) < EPS:
        print('Covariance Implementation OK')
    else:
        print('Covariance Implementation incorrect')



    


