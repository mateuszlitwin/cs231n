import numpy as np
import math
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores = scores - np.max(scores)
        sum = np.sum(np.exp(scores))
        loss_i = -math.log(math.exp(scores[y[i]]) / np.sum(np.exp(scores)))
        loss += loss_i
        for k in range(num_class):
            if k == y[i]:
                dW[:, k] += X[i] * (-1 + math.exp(scores[k]) / sum)
            else:
                dW[:, k] += X[i] * (0 + math.exp(scores[k]) / sum)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_train = X.shape[0]

    scores = X.dot(W)
    scores = (scores - np.amax(scores, axis=1, keepdims=True))
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp_scores

    loss = -np.sum(np.log(probabilities[np.arange(num_train), y]))
    loss /= num_train
    loss += reg * np.sum(W * W)

    mask = np.zeros_like(probabilities)
    mask[np.arange(num_train), y] = 1
    dW = X.T.dot(probabilities - mask)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
