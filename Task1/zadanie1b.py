import numpy as np
import pytest
from sklearn.linear_model import Ridge


class RidgeRegr:
    def __init__(self, lambd = 0.0):
        self.lambd = lambd

    def fit(self, X, Y,theta_poczatkowe,c=0.01):
        # wejscie:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Finds theta (approximately) minimizing the quadratic cost function L using an iterative method.
        n, m = X.shape
        X1= np.column_stack((np.array([1]*n),X)) # X with added column of ones

        for i in range(1,3300): # less iterations=i (1000) for  theta_poczatkowe=[0.1,0.1,0.1,0.1,0.1]
            Y_predicted = X1 @ theta_poczatkowe 
            theta_zero = np.copy(theta_poczatkowe)
            theta_zero [0] = 0
            gradient_L_= 2*X1.T @ (Y_predicted - Y) + 2 * self.lambd * theta_zero
            theta_poczatkowe = theta_poczatkowe - c * gradient_L_

        self.theta = theta_poczatkowe
        return self
    
    def predict(self, X):
        # input
        #  X = np.array, shape = (n, m)
        # returns
        #  Y = wektor(f(X_1), ..., f(X_n))
        n, m = X.shape
        return (np.column_stack((np.array([1]*n),X)) @ self.theta)


def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    lambd= 0.3 # 0.3
    expected = Ridge(lambd).fit(X, Y).predict(X_test)
    theta_poczatkowe=np.array([0,0])
    c=0.01 # precyzyjniej 0.0112
    actual = RidgeRegr(lambd).fit(X, Y,theta_poczatkowe,c).predict(X_test)
    print("one dim test: ")
    print("expected:", expected)
    print("actual:",actual,"with parameteres : \n theta_pocztkowe",theta_poczatkowe,"\n c:",c)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    lambd = 0.4
    model= Ridge(lambd).fit(X, Y)
    expected =model.predict(X_test)

    print("coefficients",model.coef_)

    theta_poczatkowe=np.array([0,0,0,0])
    c=0.005
    model2=RidgeRegr(lambd).fit(X, Y,theta_poczatkowe,c)

    print("model2.theta: ",model2.theta)
    actual = model2.predict(X_test)
    print("three dim test: ")
    print("expected:", expected)
    print("actual:",actual,"with parameteres : \n theta_pocztkowe",theta_poczatkowe,"\n c:",c)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)

#test_RidgeRegressionInOneDim()
test_RidgeRegressionInThreeDim()
