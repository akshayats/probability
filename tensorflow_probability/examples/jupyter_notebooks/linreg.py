import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

def generate_data(N):
    X = np.zeros((N,2), dtype=np.float32)
    X[:,0] = np.linspace(-1, 1, N, dtype=np.float32)
    X[:,1] = 1

    W = np.random.randn(2)
    W = W.reshape(-1,1)

    beta = np.abs(np.random.randn(1))
    beta = 1.0
    y = X.dot(W) + np.random.normal(0, beta, (N,1))

    y_tf = tf.convert_to_tensor(y,dtype=np.float32)
    X_tf = tf.convert_to_tensor(X,dtype=np.float32)
    W_tf = tf.convert_to_tensor(W,dtype=np.float32)
    beta_tf = tf.convert_to_tensor(beta,dtype=np.float32)

    return y_tf, X_tf, W_tf, beta_tf

def log_prior_W(W, std):
    rv_W = tfd.Independent(
        distribution=tfd.Normal(
            loc=0.,
            scale=std
            ),
        validate_args=True,
    )
    return tf.reduce_sum(rv_W.log_prob(W))
    
def log_prior_beta(beta, std):
    rv_beta = tfd.Normal(
        loc=0.,
        scale=std,
        validate_args=True,
    )
    return rv_beta.log_prob(beta)

def log_prior_X(X, std):
    rv_X = tfd.Independent(
        distribution=tfd.Normal(
            loc=0.,
            scale=std,
            validate_args=True,
        ),
    )
    return tf.reduce_sum(rv_X.log_prob(X))
    
def log_likelihood(y,X,W,beta):
    rv_y = tfd.Normal(
        loc=tf.matmul(X,W),
        scale=tf.sqrt(tf.reciprocal(beta)),
        validate_args=True,
    )
    return tf.reduce_sum(rv_y.log_prob(y))
        
def get_joint_log_W(y, X, W, beta, variance_X, variance_W, variance_beta):
    ''' 
    Joint model for regression
    '''
    def _joint_log_W(W):
        return log_likelihood(y,X,W,beta) + log_prior_W(W, variance_W) + log_prior_beta(beta, variance_beta)
    return _joint_log_W

def get_joint_log_W_pca(y, X, W, beta, variance_X, variance_W, variance_beta):
    '''
    Joint model for PCA
    '''
    def _joint_log_W(X, W):
        return log_likelihood(y,X,W,beta) + log_prior_X(X, variance_X) + log_prior_W(W, variance_W) + log_prior_beta(beta, variance_beta)
    return _joint_log_W

def posterior_mean_var(y_tf, X_tf, W_tf, beta_tf, std_W):
    S_0_inv = tf.matrix_inverse(
        tf.multiply(tf.eye(num_rows=2,
                           dtype=np.float32),
                    tf.square(tf.div(1.0,
                                     std_W)
                    )
        )
    )
    var = tf.matrix_inverse(S_0_inv+\
                            tf.multiply(beta_tf,
                                        tf.matmul(
                                            X_tf,X_tf,
                                            transpose_a=True,
                                            transpose_b=False
                                            )
                                        )
                            )

    mean = tf.matmul(var,
                     tf.matmul(S_0_inv,
                               tf.zeros([2,1],dtype=np.float32))+\
                     tf.matmul(X_tf,
                               y_tf,
                               transpose_a=True,
                               transpose_b=False
                               )
                     )


    return mean, var

with tf.Graph().as_default() as g:
    # generate data and init values
    y_tf, X_tf, W_tf, beta_tf = generate_data(100)
    init_W = np.random.randn(2)
    init_W = init_W.reshape(-1,1)
    init_W_tf = tf.convert_to_tensor(init_W, dtype=np.float32)

    # set-up priors
    std_X = 1.0
    std_W = 1.0
    std_beta = 1.0

    # W
    joint_log_W_fn = get_joint_log_W(y_tf, X_tf, W_tf, beta_tf,
                                     std_X, std_W,
                                     std_beta)

    W_posterior_mean, W_posterior_var = posterior_mean_var(y_tf,X_tf, W_tf,
                                                           beta_tf,
                                                           tf.convert_to_tensor(
                                                               std_W, dtype=np.float32))
    
    # sampler parameters
    num_results = 100
    num_burnin_steps = 10
    

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
            init_W_tf,
            ],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=joint_log_W_fn,
            step_size=0.01,
            num_leapfrog_steps=3,
            seed=42),
        parallel_iterations=1)
    W = states
    g.finalize()
    
with tf.Session(graph=g) as sess:
    tf.set_random_seed(42)

    [
        W_,
        is_accepted_,
        W_posterior_mean_,
        W_posterior_var_,
        W_tf_
    ] = sess.run([
        W,
        kernel_results.is_accepted,
        W_posterior_mean,
        W_posterior_var,
        W_tf
    ])

    print('Acceptance Rate: {}'.format(np.sum(is_accepted_) / num_results))
    print('True Posterior\nMean:', W_posterior_mean.eval().flatten(),
          '\nVariance:', np.diag(W_posterior_var.eval()).flatten())
    print('Estimate\nMean', np.mean(W_, axis=1).flatten(),
          '\nSample Variance:', np.var(W_, axis=1).flatten())
    print('True Weights:', W_tf.eval().flatten())
