""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    N,M = data.shape
    weights_part = weights[0:M]
    z_val = np.dot(data,weights_part) + weights[M:]


    return sigmoid(z_val).reshape(-1,1)


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    N = y.shape[0]
    #print(y.shape)
    cross_entropy = -1*targets*np.log(y) - (1-targets)*np.log(1-y)
    ce = cross_entropy.sum()
    frac_correct = ((y >= 0.5).astype(np.int) == targets).sum()*1.0/N
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """
    #print weights.shape,data.shape

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        #print targets.shape,y.shape
        cross_entropy = -1*targets*np.log(y) - (1-targets)*np.log(1-y)
        f = cross_entropy.sum()
        wj_df = np.sum(((1-targets)*data -data*(1-y)).T,axis=1)
        w0_df = np.sum((y-targets).T,axis = 1)
        df = np.append(wj_df,w0_df,axis = 0).reshape(-1,1)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)

    D = data.shape[1]

    cross_entropy = -1*targets*np.log(y) - (1-targets)*np.log(1-y)
    #print cross_entropy.shape
    f = cross_entropy.sum() + D/2*np.log(2*np.pi) - 1/2*np.log(hyperparameters['weight_decay']) + 1/2*hyperparameters['weight_decay']*np.dot(weights.T,weights).sum()
    wj_df = np.sum(((1-targets)*data -data*(1-y)).T,axis=1).reshape(-1,1)
    w0_df = np.sum((y-targets).T,axis = 1).reshape(-1,1)
    df = np.append(wj_df,w0_df,axis = 0).reshape(-1,1) + hyperparameters['weight_decay']*weights.reshape(-1,1)
    #print df.shape
    #print f.shape

    return f, df
