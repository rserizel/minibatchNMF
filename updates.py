"""
updates.py
~~~~~~~~~~
.. topic:: Contents

  The update module regroups the update functions used for
  the mini-batch NMF"""
import theano.tensor as T
from theano.ifelse import ifelse
import theano


def gradient_h(X, W, H, beta):
    """Compute the gradient of the beta-divergence relatively to the factor H

    Parameters
    ----------
    X: theano tensor
        Data matrix to be decomposed

    W: theano tensor
        Factor matrix containing the bases of the decomposition

    H: theano tensor
        Factor matrix containing the actiovations of the decomposition

    beta: theano scalar
        Coefficient beta for the beta-divergence
        Special cases:
        * beta = 1: Itakura-Saito
        * beta = 1: Kullback-Leibler
        * beta = 2: Euclidean distance

    Returns
    -------
    grad_h: theano matrix
        Gradient of the local beta-divergence with respect to H
    """
    grad_h = ifelse(T.eq(beta, 2), T.dot(X, W) - T.dot(T.dot(H, W.T), W),
                    ifelse(T.eq(beta, 1), T.dot(T.mul(T.power(T.dot(H, W.T),
                                                              (-1)), X), W) -
                           T.dot(T.ones_like(X), W),
                           T.dot(T.mul(T.power(T.dot(H, W.T),
                                               (beta - 2)), X), W) -
                           T.dot(T.power(T.dot(H, W.T), (beta-1)), W)
                           ))
    return grad_h


def gradient_h_mu(X, W, H, beta):
    """Compute the gradient of the beta-divergence relatively to the factor H
    Return positive and negative contribution e.g. for multiplicative updates

    Parameters
    ----------
    X: theano tensor
        Data matrix to be decomposed

    W: theano tensor
        Factor matrix containing the bases of the decomposition

    H: theano tensor
        Factor matrix containing the actiovations of the decomposition

    beta: theano scalar
        Coefficient beta for the beta-divergence
        Special cases:
        * beta = 1: Itakura-Saito
        * beta = 1: Kullback-Leibler
        * beta = 2: Euclidean distance

    Returns
    -------
    grad_h: theano matrix (T.stack(grad_h_pos, grad_h_neg))
        :grad_h_pos: Positive term of the gradient
          of the local beta-divergence with respect to H
        :grad_h_neg: Positive term of the gradient
          of the local beta-divergence with respect to H
    """
    grad_h_neg = ifelse(T.eq(beta, 2), T.dot(X, W),
                        ifelse(T.eq(beta, 1),
                               T.dot(T.mul(T.power(T.dot(H, W.T),
                                                   (-1)), X), W),
                               T.dot(T.mul(T.power(T.dot(H, W.T),
                                                   (beta - 2)), X), W)))
    grad_h_pos = ifelse(T.eq(beta, 2), T.dot(T.dot(H, W.T), W),
                        ifelse(T.eq(beta, 1), T.dot(T.ones_like(X), W),
                               T.dot(T.power(T.dot(H, W.T), (beta-1)), W)))
    return T.stack(grad_h_pos, grad_h_neg)


def gradient_w(X, W, H, beta):
    """Compute the gradient of the beta-divergence relatively to the factor W

    Parameters
    ----------
    X: theano tensor
        Data matrix to be decomposed

    W: theano tensor
        Factor matrix containing the bases of the decomposition

    H: theano tensor
        Factor matrix containing the actiovations of the decomposition

    beta: theano scalar
        Coefficient beta for the beta-divergence
        Special cases:
        * beta = 1: Itakura-Saito
        * beta = 1: Kullback-Leibler
        * beta = 2: Euclidean distance

    Returns
    -------
    grad_w: theano matrix
        Gradient of the local beta-divergence with respect to W
    """
    grad_w = ifelse(T.eq(beta, 2), T.dot(X.T, H) - T.dot(T.dot(H, W.T).T, H),
                    ifelse(T.eq(beta, 1), T.dot(T.mul(T.power(T.dot(H, W.T),
                                                              (-1)), X).T, H) -
                           T.dot(T.ones_like(X).T, H),
                           T.dot(T.mul(T.power(T.dot(H, W.T),
                                               (beta - 2)), X).T, H) -
                           T.dot(T.power(T.dot(H, W.T), (beta-1)).T, H)
                           ))
    return grad_w


def gradient_w_mu(X, W, H, beta):
    """Compute the gradient of the beta-divergence relatively to the factor W
    Return positive and negative contribution e.g. for multiplicative updates

    Parameters
    ----------
    X: theano tensor
        Data matrix to be decomposed

    W: theano tensor
        Factor matrix containing the bases of the decomposition

    H: theano tensor
        Factor matrix containing the actiovations of the decomposition

    beta: theano scalar
        Coefficient beta for the beta-divergence
        Special cases:
        * beta = 1: Itakura-Saito
        * beta = 1: Kullback-Leibler
        * beta = 2: Euclidean distance

    Returns
    -------
    grad_w: theano matrix (T.stack(grad_w_pos, grad_w_neg))
        :grad_w_pos: Positive term of the gradient
          of the local beta-divergence with respect to W
        :grad_w_neg: Positive term of the gradient
          of the local beta-divergence with respect to W
    """
    grad_w_neg = ifelse(T.eq(beta, 2), T.dot(X.T, H),
                        ifelse(T.eq(beta, 1),
                               T.dot(T.mul(T.power(T.dot(H, W.T),
                                                   (-1)), X).T, H),
                               T.dot(T.mul(T.power(T.dot(H, W.T),
                                                   (beta - 2)), X).T, H)))
    grad_w_pos = ifelse(T.eq(beta, 2), T.dot(T.dot(H, W.T).T, H),
                        ifelse(T.eq(beta, 1), T.dot(T.ones_like(X).T, H),
                               T.dot(T.power(T.dot(H, W.T), (beta-1)).T, H)))
    return T.stack(grad_w_pos, grad_w_neg)


def mu_update_h(X, W, H, beta):
    """Compute the gradient of the beta-divergence relatively to the factor H
       and update H with multiplicative rules

    Parameters
    ----------
    X: theano tensor
        Data matrix to be decompsed

    W: theano tensor
        Factor matrix containing the bases of the decomposition

    H: theano tensor
        Factor matrix containing the activations of the decomposition

    beta: theano scalar
        Coefficient beta for the beta-divergence
        Special cases:
        * beta = 1: Itakura-Saito
        * beta = 1: Kullback-Leibler
        * beta = 2: Euclidean distance


    Returns
    -------
    H: theano matrix
        New value of H updated with multiplicative updates
    """
    grad_h_neg = ifelse(T.eq(beta, 2), T.dot(X, W),
                        ifelse(T.eq(beta, 1),
                               T.dot(T.mul(T.power(T.dot(H, W.T),
                                                   (-1)), X), W),
                               T.dot(T.mul(T.power(T.dot(H, W.T),
                                                   (beta - 2)), X), W)))
    grad_h_pos = ifelse(T.eq(beta, 2), T.dot(T.dot(H, W.T), W),
                        ifelse(T.eq(beta, 1), T.dot(T.ones_like(X), W),
                               T.dot(T.power(T.dot(H, W.T), (beta-1)), W)))
    H *= grad_h_neg/grad_h_pos
    return H, T.stack(grad_h_pos, grad_h_neg)


def mu_update(factor, gradient_pos, gradient_neg):
    """Update the factor based on multiplicative rules

    Parameters
    ----------
    factor: theano tensor
        The factor to be updated

    gradient_pos: theano tensor
        Positive part of gradient relatively to factor

    gradient_neg: theano tensor
        Negative part of gradient relatively to factor

    Returns
    -------
    factor: theano matrix
        New value of factor update with multiplicative updates
    """
    factor *= gradient_neg/gradient_pos
    return factor


def update_grad_w(grad, grad_old, grad_new):
    """Update the global gradient for W

    Parameters
    ----------
    grad: theano tensor
        The global gradient

    grad_old: theano tensor
        The previous value of the local gradient

    grad_new: theano tensor
        The new version of the local gradient

    Returns
    -------
    grad: theano tensor
        New value of the global gradient
    """
    return grad - grad_old + grad_new
