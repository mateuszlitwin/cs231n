"""
xx
"""

from __future__ import print_function

import numpy as np


def softmax(w):
    scores = w - np.amax(w, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp_scores
    return probabilities


class TwoLayerNet:
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, input, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        weights1, bias1 = self.params['W1'], self.params['b1']
        weights2, bias2 = self.params['W2'], self.params['b2']

        forward1 = input
        forward2 = forward1.dot(weights1) + bias1.reshape(1, weights1.shape[1])
        forward2 = np.maximum(0, forward2)
        forward3 = forward2.dot(weights2) + bias2.reshape(1, weights2.shape[1])

        if y is None:
            return forward3
        forward3 = softmax(forward3)

        backward3 = forward3.copy()
        backward3[np.arange(forward1.shape[0]), y] -= 1
        backward3 /= forward1.shape[0]
        backward2 = backward3.dot(weights2.T)
        backward2[forward2 == 0] = 0

        grads = {}

        data_loss = -np.sum(np.log(forward3[np.arange(forward1.shape[0]), y] + 1e-6))
        data_loss /= forward1.shape[0]
        grads['W2'] = forward2.T.dot(backward3)
        grads['b2'] = np.sum(backward3, axis=0)
        grads['W1'] = forward1.T.dot(backward2)
        grads['b1'] = np.sum(backward2, axis=0)

        reg_loss = 0
        reg_loss += 0.5 * reg * np.sum(weights2 * weights2)
        reg_loss += 0.5 * reg * np.sum(weights1 * weights1)
        grads['W2'] += reg * weights2
        grads['W1'] += reg * weights1

        return data_loss + reg_loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            idxs = np.random.randint(X.shape[0], size=batch_size)
            X_batch = X[idxs]
            y_batch = y[idxs]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % (iterations_per_epoch) == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        weights1, bias1 = self.params['W1'], self.params['b1']
        weights2, bias2 = self.params['W2'], self.params['b2']

        forward1 = X
        forward2 = forward1.dot(weights1) + bias1.reshape(1, weights1.shape[1])
        forward2 = np.maximum(0, forward2)
        forward3 = forward2.dot(weights2) + bias2.reshape(1, weights2.shape[1])

        y_pred = np.argmax(forward3, axis=1)
        return y_pred
