import numpy as np
import pytest
from sklearn.linear_model import LinearRegression


class LinearRegr:
    def fit(self, X, Y):
        # wejscie:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Znajduje theta=beta minimalizujace kwadratowa funkcje kosztu L uzywajac wzoru.
        # Uwaga: przed zastosowaniem wzoru do X nalezy dopisac kolumne zlozona z jedynek.
        n, m = X.shape
        #self.theta = np.zeros((m+1))
        # np.ones(n)=np.array([1]*n)
        X1= np.column_stack((np.array([1]*n),X)) # X with added column of ones
        X1T_X1=X1.T @ X1

        #checking if X1^T*X1 is invertible 
        k,l=X1T_X1.shape
        if k==l and np.linalg.det(X1T_X1)!=0:
            X1T_X1_inverse=np.linalg.inv(X1T_X1)
        else:
            raise ValueError("Matrix X1T_X1 is not invertible")
        
        self.theta =  (X1T_X1_inverse @ X1.T) @ Y

        return self
    

    def predict(self, X):
        # wejscie
        #  X = np.array, shape = (n, m)
        # zwraca
        #  Y = wektor(f(X_1), ..., f(X_n))
        n,m = X.shape
        return (np.column_stack((np.array([1]*n),X)) @ self.theta)


def test_RegressionInOneDim(): 
    X = np.array([1,3,2,5]).reshape((4,1)) #training data
    Y = np.array([2,5,3,8]) 
    a = np.array([1,2,10]).reshape((3,1)) # validation data
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print("one dim test: ")
    print("expected:", expected)
    print("actual:",actual)
    assert list(actual) == pytest.approx(list(expected)),"error in one dimension function"

def test_RegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print("three dim test: ")
    print("expected:", expected)
    print("actual:",actual)
    assert list(actual) == pytest.approx(list(expected)), "error in three dimension function"

test_RegressionInOneDim()
test_RegressionInThreeDim()

    
