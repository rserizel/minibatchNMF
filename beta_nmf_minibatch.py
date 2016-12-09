# -*- coding: utf-8 -*-
"""
beta\_nmf_minibatch.py
~~~~~~~~~~~

.. topic:: Contents

  The beta_nmf_minibatch module includes the betaNMF class,
  fit function and theano functions to compute updates and cost."""

import time
import numpy as np
import theano
import base
import theano.tensor as T
import updates
import costs


class BetaNMF(object):
    """BetaNMF class

    Performs nonnegative matrix factorization with mini-batch multiplicative
    updates. GPGPU implementation based on Theano.

    Parameters
    ----------
    data_shape : tuple composed of integers
        The shape of the data to approximate

    n_components : positive integer
        The number of latent components for the NMF model


    beta: arbitrary float (default 2).
        The beta-divergence to consider. Particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    n_iter: positive integer
         number of iterations

    fixed_factors: array of intergers
        Indexes of the factors that are kept fixed during the updates
        * [0] : corresponds to fixed H
        * [1] : corresponds to fixed W

    cache1_size: integer
        Size (in frames) of the first data cache.
        The size is reduced to the closest multiple of the batch_size.
        If set to zero the algorithm tries to fit all the data in cache

    batch_size: integer
        Size (in frames) of the batch for batch processing.
        The batch size has an impact on the parrelization and the memory needed
        to store partial gradients (see Schmidt et al.)

    verbose: integer
        The numer of iterations to wait between two computation and printing
        of the cost

    init_mode : string (default 'random')
        * random : initalise the factors randomly
        * custom : intialise the factors with custom value

    W : array (optionnal)
        Initial wvalue for factor W when custom initialisation is used

    H : array (optionnal)
        Initial wvalue for factor H when custom initialisation is used

    solver : string (default 'mu_batch')
        * mu_batch : mini-batch version of the MU updates.
         (fully equivalent to standard NMF with MU).
        * asg_mu : Asymetric stochatistic gradient for MU [1]_
        * gsg_mu : Greedy stochatistic gradient for  MU [1]_
        * asag_mu : Asymetric stochatistic average gradient [2]_ for MU [1]_
        * gsag_mu : Greedy stochatistic average gradient [2]_ for MU [1]_

    nb_batch_w : interger (default 1)
        number of batches on which W updates is computed
        * 1 : greedy approaches [1]_

    sag_memory : integer (default 0)
        number of batches used to compute the average gradient
        * 0 : SG approaches
        * nb_batches : SAG approaches

    Attributes
    ----------
    nb_cache1 : integer
        number of caches needed to fill the full data

    forget_factor : float
        forgetting factor for SAG

    scores : array
        reconstruction cost and iteration time for each iteration

    factors\_ : list of arrays
        The estimated factors

    w : theano tensor
        factor W

    h_cache1 : theano tensor
        part of the factor H in cache1

    x_cache1 : theano tensor
        data cache

    References
    ----------
    .. [#] R. Serizel, S. Essid, and G. Richard. “Mini-batch stochastic
        approaches for accelerated multiplicative updates in nonnegative matrix
        factorisation with beta-divergence”. Accepted for publication
        In *Proc. of MLSP*, p. 5, 2016.

    .. [#] Schmidt, M., Roux, N. L., & Bach, F. (2013).
        Minimizing finite sums with the stochastic average gradient
        https://hal.inria.fr/hal-00860051/PDF/sag_journal.pdf
    """

    # Constructor
    def __init__(self, data_shape, n_components=50, beta=2, n_iter=50,
                 fixed_factors=None, cache1_size=0,
                 batch_size=100, verbose=0,
                 init_mode='random', W=None, H=None, solver='mu_batch',
                 nb_batch_w=1, sag_memory=0):
        self.data_shape = data_shape
        self.n_components = n_components
        self.batch_size = batch_size
        self.nb_batch = int(np.ceil(np.true_divide(data_shape[0],
                                                   self.batch_size)))
        self.batch_ind = np.zeros((self.nb_batch, self.batch_size))

        if cache1_size > 0:
            cache1_size = min((cache1_size, data_shape[0]))
            if cache1_size < self.batch_size:
                raise ValueError('cache1_size should be at '
                                 'least equal to batch_size')
            self.cache1_size = cache1_size/self.batch_size * self.batch_size
            self.nb_cache1 = int(np.ceil(np.true_divide(self.data_shape[0],
                                                        self.cache1_size)))
        else:
            self.cache1_size = data_shape[0]
            self.nb_cache1 = 1

        self.n_components = np.asarray(n_components, dtype='int32')
        self.beta = theano.shared(np.asarray(beta, theano.config.floatX),
                                  name="beta")
        self.eps = theano.shared(np.asarray(1e-10, theano.config.floatX),
                                  name="eps")
        self.sag_memory = sag_memory
        self.forget_factor = 1./(self.sag_memory + 1)
        self.verbose = verbose
        self.n_iter = n_iter
        self.solver = solver
        self.scores = []
        self.nb_batch_w = nb_batch_w
        if fixed_factors is None:
            fixed_factors = []
        self.fixed_factors = fixed_factors
        fact_ = [base.nnrandn((dim, self.n_components)) for dim in data_shape]
        self.init_mode = init_mode
        if self.init_mode == 'custom':
            fact_[0] = H
            fact_[1] = W
        self.w = theano.shared(fact_[1].astype(theano.config.floatX),
                               name="W", borrow=True, allow_downcast=True)
        self.h_cache1 = theano.shared(fact_[0][:self.cache1_size,
                                               ].astype(theano.config.floatX),
                                      name="H cache1", borrow=True,
                                      allow_downcast=True)
        self.factors_ = fact_
        self.x_cache1 = theano.shared(np.zeros((self.cache1_size,
                                                data_shape[1])).astype(
            theano.config.floatX),
            name="X cache1")
        self.init()

    def check_shape(self):
        """Check that all the matrix have consistent shapes
        """
        batch_shape = self.x_cache1.get_value().shape
        dim = long(self.n_components)
        if self.w.get_value().shape != (self.data_shape[1], dim):
            print "Inconsistent data for W, expected {1}, found {0}".format(
                self.w.get_value().shape,
                (self.data_shape[1], dim))
            raise SystemExit
        if self.factors_[0].shape != (self.data_shape[0], dim):
            print "Inconsistent shape for H, expected {1}, found {0}".format(
                self.factors_[0].shape,
                (self.data_shape[0], dim))
            raise SystemExit
        if self.h_cache1.get_value().shape != (batch_shape[0], dim):
            print "Inconsistent shape for h_cache1, expected {1}, found {0}".format(
                self.h_cache1.get_value().shape,
                (batch_shape[0], dim))
            raise SystemExit

    def fit(self, data, cyclic=False, warm_start=False):
        """Learns NMF model

        Parameters
        ----------
        data : ndarray with nonnegative entries
            The input array

        cyclic : Boolean (default False)
            pick the sample cyclically

        warm_start : Boolean (default False)
            start from previous values
        """
        self.data_shape = data.shape
        if (not warm_start) & (self.init_mode is not 'custom'):
            print "cold start"
            self.set_factors(data, fixed_factors=self.fixed_factors)
        self.check_shape()
        self.prepare_batch(False)
        self.prepare_cache1(False)
        div_func = self.get_div_function()
        if self.verbose > 0:
            scores = np.zeros((
                int(np.floor(self.n_iter/self.verbose)) + 2, 2))
        else:
            scores = np.zeros((2, 2))
        if self.solver is 'asag_mu' or self.solver is 'gsag_mu':
            grad_func = self.get_gradient_mu_sag()
            update_func = self.get_updates()
        elif self.solver is 'asg_mu' or self.solver is 'gsg_mu':
            grad_func = self.get_gradient_mu_sg()
            update_func = self.get_updates()
        elif self.solver is 'mu_batch':
            grad_func = self.get_gradient_mu_batch()
            update_func = self.get_updates()
        tick = time.time()
        score = 0

        for cache_ind in range(self.nb_cache1):
            current_cache_ind = np.hstack(self.batch_ind[
                    self.cache1_ind[
                        cache_ind, self.cache1_ind[cache_ind] >= 0]])
            current_cache_ind = current_cache_ind[current_cache_ind >= 0]
            self.x_cache1.set_value(data[current_cache_ind, ].astype(
                    theano.config.floatX))
            self.h_cache1.set_value(self.factors_[0][
                    current_cache_ind, ].astype(theano.config.floatX))
            score += div_func['div_cache1']()
        score_ind = 0
        scores[0, ] = [score, time.time() - tick]

        self.prepare_batch(not cyclic)
        self.prepare_cache1(not cyclic)

        print 'Intitial score = %.2f' % score
        print 'Fitting NMF model with %d iterations....' % self.n_iter
        if self.nb_cache1 == 1:
            current_cache_ind = np.hstack(self.batch_ind[
                self.cache1_ind[
                    0, self.cache1_ind[0] >= 0]])
            current_cache_ind = current_cache_ind[current_cache_ind >= 0]
            self.x_cache1.set_value(data[current_cache_ind, ].astype(
                theano.config.floatX))
            self.h_cache1.set_value(self.factors_[0][
                current_cache_ind, ].astype(theano.config.floatX))
            if self.solver is 'sag':
                self.c1_grad_w.set_value(self.old_grad_w[self.cache1_ind[
                        0, self.cache1_ind[0] >= 0]].astype(
                    theano.config.floatX))
        # main loop
        for it in range(self.n_iter):
            tick = time.time()
            self.prepare_cache1(not cyclic)
            score = 0
            for cache_ind in range(self.nb_cache1):
                if self.nb_cache1 > 1:
                    current_cache_ind = np.hstack(self.batch_ind[
                        self.cache1_ind[
                            cache_ind, self.cache1_ind[cache_ind] >= 0]])
                    current_cache_ind = current_cache_ind[
                        current_cache_ind >= 0]
                    self.x_cache1.set_value(data[current_cache_ind, ].astype(
                        theano.config.floatX))

                    self.h_cache1.set_value(self.factors_[0][
                        current_cache_ind, ].astype(theano.config.floatX))
                if self.solver is 'sag':
                    self.c1_grad_w.set_value(
                        self.old_grad_w[
                            self.cache1_ind[
                                cache_ind,
                                self.cache1_ind[cache_ind] >= 0]].astype(
                            theano.config.floatX))
                for batch_i in range(self.cache1_ind[
                        cache_ind, self.cache1_ind[cache_ind] >= 0].shape[0]):
                    batch_ind = np.arange(batch_i * self.batch_size,
                                          (batch_i + 1) * self.batch_size)
                    batch_ind = batch_ind[
                        batch_ind < current_cache_ind.shape[0]]
                    batch_ind = np.asarray([batch_ind[0],
                                            batch_ind[-1] + 1]).astype(
                        theano.config.floatX)

                    if self.solver is 'mu_batch':
                        self.update_mu_batch_h(batch_ind,
                                               update_func, grad_func)
                    if self.solver is 'asag_mu' or self.solver is 'asg_mu':
                        self.update_mu_sag(batch_ind,
                                           update_func, grad_func)
                    if self.solver is 'gsag_mu' or self.solver is 'gsg_mu':
                        grad_func['grad_h'](batch_ind)
                        update_func['train_h'](batch_ind)
                        if batch_i == 0 and cache_ind == 0:
                            grad_func['grad_w'](batch_ind)
                if self.nb_cache1 > 1:
                    self.factors_[0][current_cache_ind, ] =\
                        self.h_cache1.get_value()
                else:
                    self.factors_[0] = self.h_cache1.get_value()
            if self.solver is 'mu_batch':
                self.update_mu_batch_w(update_func)
            elif self.solver is 'gsag_mu' or self.solver is 'gsg_mu':
                update_func['train_w']()
            if self.nb_cache1 > 1:
                for cache_ind in range(self.nb_cache1):
                    self.x_cache1.set_value(data[np.hstack(self.batch_ind[
                        self.cache1_ind[
                            cache_ind,
                            self.cache1_ind[cache_ind] >= 0]]), ].astype(
                        theano.config.floatX))
                    self.h_cache1.set_value(self.factors_[0][
                        np.hstack(self.batch_ind[
                            self.cache1_ind[
                                cache_ind, self.cache1_ind[cache_ind] >= 0]]),
                        ].astype(theano.config.floatX))
                    if (it+1) % self.verbose == 0:
                        score += div_func['div_cache1']()
            else:
                self.factors_[0] = self.h_cache1.get_value()
                if (it+1) % self.verbose == 0:
                    score = div_func['div_cache1']()
            if (it+1) % self.verbose == 0:
                score_ind += 1
                scores[score_ind, ] = [
                    score, time.time() - tick + scores[score_ind - 1, 1]]
                print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                       % (it + 1,
                          self.n_iter,
                          scores[score_ind, 1] * 1000,
                          scores[score_ind, 0]))
                tick = time.time()
        score_ind += 1
        scores[score_ind, ] = [
            score, time.time() - tick + scores[score_ind - 1, 1]]
        print ('Iteration %d / %d, duration=%.1fms, cost=%f'
               % (it + 1,
                  self.n_iter,
                  scores[-1, 1] * 1000,
                  scores[-1, 0]))        
        return scores

    def get_div_function(self):
        """ compile the theano-based divergence function"""
        div_cache1 = theano.function(inputs=[],
                                     outputs=costs.beta_div(self.x_cache1,
                                                            self.w.T,
                                                            self.h_cache1,
                                                            self.beta),
                                     name="div c1",
                                     allow_input_downcast=True, profile=False)
        return dict(
            div_cache1=div_cache1)

    def get_gradient_mu_sag(self):
        """compile the theano based gradient functions for mu_sag algorithms"""
        tbatch_ind = T.ivector('batch_ind')
        tind = T.iscalar('ind')
        grad_new = updates.gradient_w_mu(
                                    self.x_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.w,
                                    self.h_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.beta)
        up_grad_w = self.forget_factor * grad_new + (
            1 - self.forget_factor) * self.grad_w

        grad_w = theano.function(inputs=[tbatch_ind],
                                 outputs=[],
                                 updates={(self.grad_w, up_grad_w)},
                                 name="grad w",
                                 allow_input_downcast=True)
        grad_new = updates.gradient_h_mu(
                                    self.x_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.w,
                                    self.h_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.beta)

        grad_h = theano.function(inputs=[tbatch_ind],
                                 outputs=[],
                                 updates={(self.c1_grad_h, grad_new)},
                                 name="grad h",
                                 allow_input_downcast=True)
        return dict(
            grad_h=grad_h,
            grad_w=grad_w)

    def get_gradient_mu_sg(self):
        """compile the theano based gradient functions for mu_sg algorithms"""
        tbatch_ind = T.ivector('batch_ind')
        tind = T.iscalar('ind')
        grad_new = updates.gradient_w_mu(
                                    self.x_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.w,
                                    self.h_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.beta)

        grad_w = theano.function(inputs=[tbatch_ind],
                                 outputs=[],
                                 updates={(self.grad_w, grad_new)},
                                 name="grad w",
                                 allow_input_downcast=True)
        grad_new = updates.gradient_h_mu(
                                    self.x_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.w,
                                    self.h_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.beta)

        grad_h = theano.function(inputs=[tbatch_ind],
                                 outputs=[],
                                 updates={(self.c1_grad_h, grad_new)},
                                 name="grad h",
                                 allow_input_downcast=True)
        return dict(
            grad_h=grad_h,
            grad_w=grad_w)

    def get_gradient_mu_batch(self):
        """compile the theano based gradient functions for mu"""
        tbatch_ind = T.ivector('batch_ind')
        tind = T.iscalar('ind')
        grad_new = updates.gradient_w_mu(
                                    self.x_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.w,
                                    self.h_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.beta)

        grad_w = theano.function(inputs=[tbatch_ind],
                                 outputs=[],
                                 updates={(self.grad_w,
                                           self.grad_w + grad_new)},
                                 name="grad w",
                                 allow_input_downcast=True,
                                 on_unused_input='ignore')
        grad_new = updates.gradient_h_mu(
                                    self.x_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.w,
                                    self.h_cache1[tbatch_ind[0]:tbatch_ind[1],
                                                  ],
                                    self.beta)
        grad_h = theano.function(inputs=[tbatch_ind],
                                 outputs=[],
                                 updates={(self.c1_grad_h, grad_new)},
                                 name="grad h",
                                 allow_input_downcast=True)
        return dict(
            grad_h=grad_h,
            grad_w=grad_w)

    def get_updates(self):
        """compile the theano based update functions"""
        tbatch_ind = T.ivector('batch_ind')
        tneg = T.iscalar('neg')
        tpos = T.iscalar('pos')
        up_h = T.set_subtensor(self.h_cache1[tbatch_ind[0]:tbatch_ind[1], ],
                               updates.mu_update(self.h_cache1[
                                    tbatch_ind[0]:tbatch_ind[1], ],
                                    self.c1_grad_h[0, ],
                                    self.c1_grad_h[1, ],
                                    self.eps))
        train_h = theano.function(inputs=[tbatch_ind],
                                  outputs=[],
                                  updates={(self.h_cache1, up_h)},
                                  name="trainH",
                                  allow_input_downcast=True,
                                  on_unused_input='ignore')
        update_w = updates.mu_update(self.w,
                                     self.grad_w[0],
                                     self.grad_w[1],
                                     self.eps)
        train_w = theano.function(inputs=[],
                                  outputs=[],
                                  updates={self.w: update_w},
                                  name="trainW",
                                  allow_input_downcast=True)
        return dict(
            train_h=train_h,
            train_w=train_w)

    def init(self):
        """Initialise theano variable to store the gradients"""
        self.grad_w = theano.shared(
            np.zeros((2,
                      self.data_shape[1],
                      self.n_components)).astype(theano.config.floatX),
            name="gradW", borrow=True,
            allow_downcast=True)
        self.grad_h = np.zeros((2, self.data_shape[0], self.n_components))
        self.c1_grad_h = theano.shared(
            np.zeros((2,
                      self.batch_size,
                      self.n_components)).astype(theano.config.floatX),
            name="c1_gradH", borrow=True,
            allow_downcast=True)

    def prepare_batch(self, randomize=True):
        """Arrange data for batches

        Parameters
        ----------
        randomize : boolean (default True)
            Randomise the data (time-wise) before preparing batch indexes
        """
        ind = - np.ones((self.nb_batch * self.batch_size, ))
        ind[:self.data_shape[0], ] = np.arange(self.data_shape[0])
        if randomize:
            np.random.shuffle(ind[:self.data_shape[0], ])
        self.batch_ind = np.reshape(ind, (self.nb_batch,
                                          self.batch_size)).astype(int)

    def prepare_cache1(self, randomize=True):
        """Arrange data for to fill cache1

        Parameters
        ----------
        randomize : boolean (default True)
            Randomise the data (time-wise) before preparing cahce indexes
        """
        ind = - np.ones((self.nb_cache1 *
                         int(np.ceil(np.true_divide(self.cache1_size,
                                                    self.batch_size)))))
        ind[:self.nb_batch, ] = np.arange(self.nb_batch)
        if randomize:
            np.random.shuffle(ind[:self.nb_batch, ])
        self.cache1_ind = np.reshape(ind, (self.nb_cache1,
                                           int(np.ceil(np.true_divide(
                                                self.cache1_size,
                                                self.batch_size)))
                                           )).astype(int)

    def set_factors(self, data, W=None, H=None, fixed_factors=None):
        """Re-set theano based parameters according to the object attributes.

        Parameters
        ----------
        W : array (optionnal)
            Value for factor W when custom initialisation is used

        H : array (optionnal)
            Value for factor H when custom initialisation is used

        fixed_factors : array  (default Null)
            list of factors that are not updated
                e.g. fixed_factors = [0] -> H is not updated

                fixed_factors = [1] -> W is not updated
        """
        self.data_shape = data.shape
        self.nb_batch = int(np.ceil(np.true_divide(self.data_shape[0],
                                                   self.batch_size)))
        self.batch_ind = np.zeros((self.nb_batch, self.batch_size))

        if self.cache1_size > 0 and self.cache1_size < self.data_shape[0]:
            if self.cache1_size < self.batch_size:
                raise ValueError('cache1_size should be at '
                                 'least equal to batch_size')
            self.cache1_size = self.cache1_size/self.batch_size * self.batch_size
            self.nb_cache1 = int(np.ceil(np.true_divide(self.data_shape[0],
                                                        self.cache1_size)))
        else:
            self.cache1_size = self.data_shape[0]
            self.nb_cache1 = 1

        self.forget_factor = 1./(self.sag_memory + 1)
        fact_ = [base.nnrandn((dim, self.n_components))
                 for dim in self.data_shape]
        if H is not None:
            fact_[0] = H
        if W is not None:
            fact_[1] = W
        if fixed_factors is None:
            fixed_factors = []
        if 1 not in fixed_factors:
            self.w = theano.shared(fact_[1].astype(theano.config.floatX),
                                   name="W", borrow=True, allow_downcast=True)
        if 0 not in fixed_factors:
            self.h_cache1 = theano.shared(
                fact_[0][
                    :self.cache1_size, ].astype(theano.config.floatX),
                name="H cache1", borrow=True,
                allow_downcast=True)
            self.factors_[0] = fact_[0]
        self.factors_ = fact_
        self.x_cache1 = theano.shared(np.zeros((self.cache1_size,
                                                self.data_shape[1])).astype(
            theano.config.floatX),
            name="X cache1")
        self.init()

    def transform(self, data, warm_start=False):
        """Project data X on the basis W

        Parameters
        ----------
        X : array
            The input data
        warm_start : Boolean (default False)
            start from previous values

        Returns
        -------
        H : array
            Activations
        """
        self.fixed_factors = [1]
        if not warm_start:
            print "cold start"
            self.set_factors(data, fixed_factors=self.fixed_factors)
        self.fit(data, warm_start=True)
        return self.factors_[0]

    def update_mu_sag(self, batch_ind, update_func, grad_func):
        """Update current batch with SAG based algorithms

        Parameters
        ----------
        batch_ind : array with 2 elements
            :batch_ind[0]: batch start
            :batch_ind[1]: batch end

        update_func : Theano compiled function
            Update function

        grad_func : Theano compiled function
            Gradient function
        """
        if 1 not in self.fixed_factors:
            grad_func['grad_h'](batch_ind)
            update_func['train_h'](batch_ind)
        if 0 not in self.fixed_factors:
            grad_func['grad_w'](batch_ind)
            update_func['train_w']()

    def update_mu_batch_h(self, batch_ind, update_func, grad_func):
        """Update h for current batch with standard MU

        Parameters
        ----------
        batch_ind : array with 2 elements
            :batch_ind[0]: batch start
            :batch_ind[1]: batch end

        update_func : Theano compiled function
            Update function

        grad_func : Theano compiled function
            Gradient function
        """
        if 0 not in self.fixed_factors:
            grad_func['grad_h'](batch_ind)
            update_func['train_h'](batch_ind)
            grad_func['grad_w'](batch_ind)

    def update_mu_batch_w(self, udpate_func):
        """Update W with standard MU

        Parameters
        ----------
        update_func : Theano compiled function
            Update function
        """
        if 1 not in self.fixed_factors:
            udpate_func['train_w']()
            self.grad_w.set_value(
                np.zeros((
                    2,
                    self.data_shape[1],
                    self.n_components)).astype(
                        theano.config.floatX))
