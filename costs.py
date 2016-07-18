# -*- coding: utf-8 -*-
"""
cost.py
~~~~~~~
.. topic:: Contents

    The cost module regroups the cost functions used for the group NMF"""
import theano.tensor as T
from theano.ifelse import ifelse
import theano


def beta_div(X, W, H, beta):
    """Compute beta divergence D(X|WH)

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar


    Returns
    -------
    div : Theano scalar
        beta divergence D(X|WH)"""
    div = ifelse(
      T.eq(beta, 2),
      T.sum(1. / 2 * T.power(X - T.dot(H, W), 2)),
      ifelse(
        T.eq(beta, 0),
        T.sum(X / T.dot(H, W) - T.log(X / T.dot(H, W)) - 1),
        ifelse(
          T.eq(beta, 1),
          T.sum(T.mul(X, (T.log(X) - T.log(T.dot(H, W)))) + T.dot(H, W) - X),
          T.sum(1. / (beta * (beta - 1.)) * (T.power(X, beta) +
                (beta - 1.) * T.power(T.dot(H, W), beta) -
                beta * T.power(T.mul(X, T.dot(H, W)), (beta - 1)))))))
    return div
