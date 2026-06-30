import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N = X.shape[0]
    w=np.zeros(X.shape[1])
    b=0.0
    for i in range(steps):
        y_=_sigmoid(X@w+b)
        error = y_ - y
        dw = (X.T @ error) / N
        db = np.mean(error)
        w=w-lr*dw
        b=b-lr*db

    return w , b