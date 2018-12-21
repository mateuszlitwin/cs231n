from cs231n.classifiers.linear_svm import svm_loss_naive
import numpy as np

W = np.matrix('1 2; 3 4; -1 2')
x0 = np.array([1, -2])
x1 = np.array([4, -3])
x2 = np.array([-1, 3])
x3 = np.array([-3, 5])
X = np.vstack((x0, x1, x2, x3)).T
y = np.asarray([0, 0, 1, 1])

W = W.T
X = X.T

########################################


loss, grad = svm_loss_naive(W, X, y, 0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
from cs231n.gradient_check import grad_check_sparse

f = lambda w: svm_loss_naive(w, X, y, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)
