from numpy import size, zeros, log, sum
from sigmoid import sigmoid

# COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters.

def costFunctionReg(theta, X, y, _lambda):

    # Initialize some useful values
    m = size(y, 0)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = zeros(size(theta, 0))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    htheta = sigmoid(X @ theta).T

    J = 1 / m * sum(-y * log(htheta) - (1 - y) * log(1 - htheta)) + _lambda / (2 * m) * sum(theta[2:] ** 2)

    grad[0] = 1 / m * sum((htheta - y) * X[:, 0])
    for i in range(1, size(theta, 0)):
        grad[i] = 1 / m * sum((htheta - y) * X[:, i]) + _lambda / m * theta[i]

    return J, grad

    # =============================================================
