"""Simulate a stochastic process, add noise, and try to denoise using
a GRU. Source: `https://roberttlange.github.io/posts/2020/03/blog-post-10/`. """

import argparse, time
import matplotlib.pyplot as plt
from functools import partial
import numpy as onp
import jax.numpy as np
from jax import jit, vmap, value_and_grad, random
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal
from jax import lax
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu


def generate_ou_process(n_trials, n, mu, tau, sigma, noise_std, dt):
    """Generate an Ornstein-Uhlenbeck process, and add Gaussian
    noise. Source for the OU process:
    https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/.

    Args:
    - n_trials: number of independent simulations of OU process
    - n: number of time steps for OU process
    - mu: mean for OU process
    - tau: time constant for OU process
    - sigma: standard deviation for OU process
    - noise_std: standard deviation for noise
    - dt: time step for OU process

    """
    T = dt * n  # Total time.
    t = onp.linspace(0., T, n)  # Vector of times.

    #  Define renormalized variables (to avoid recomputing these constants
    #  at every time step
    sigma_bis = sigma * onp.sqrt(2. / tau)
    sqrtdt = onp.sqrt(dt)

    # Simulate process in a vectorized way
    X = onp.zeros((n_trials, n))
    xi = onp.zeros(n_trials)
    for i in range(n):
        # Update the process independently for all trials
        xi += dt * (-(xi - mu) / tau) + sigma_bis * sqrtdt * onp.random.randn(n_trials)
        X[:,i] = onp.copy(xi)

    # Add noise
    X_tilde = X + onp.random.normal(0, noise_std, X.shape)
    return X, X_tilde


def GRU(out_dim, W_init=glorot_normal(), b_init=normal()):
    def init_func(rng, input_shape):
        """ Initialize the GRU layer for stax """
        hidden = b_init(rng, (input_shape[0], out_dim))

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)
        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], input_shape[1], out_dim)
        return (output_shape,
                (hidden,
                 (update_W, update_U, update_b),
                 (reset_W, reset_U, reset_b),
                 (out_W, out_U, out_b),),)

    def apply_func(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = params[0]

        def apply_func_scan(params, hidden, inp):
            """ Perform single step update of the network """
            _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
                out_W, out_U, out_b) = params

            update_gate = sigmoid(np.dot(inp, update_W) +
                                  np.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(np.dot(inp, reset_W) +
                                 np.dot(hidden, reset_U) + reset_b)
            output_gate = np.tanh(np.dot(inp, out_W)
                                  + np.dot(np.multiply(reset_gate, hidden), out_U)
                                  + out_b)
            output = np.multiply(update_gate, hidden) + np.multiply(1-update_gate, output_gate)
            hidden = output
            return hidden, hidden

        # Move the time dimension to position 0
        inputs = np.moveaxis(inputs, 1, 0)
        f = partial(apply_func_scan, params)
        _, h_new = lax.scan(f, h, inputs)
        return h_new
    return init_func, apply_func


def main(make_plots=False):
    key = random.PRNGKey(42)

    # Make some data
    x_0, mu, tau, sigma, dt = 0, 1, 2, 0.5, 0.1
    noise_std = 0.2
    num_steps, num_trials = 100, 50
    x, x_tilde = generate_ou_process(num_trials, num_steps, mu, tau, sigma, noise_std, dt)
    if make_plots:
        plt.plot(x[0,:], linewidth=2, color="blue", label="clean")
        plt.plot(x_tilde[0,:], linewidth=2, color="orange", label="noisy")
        plt.title("Ornstein-Uhlenbeck Process")
        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        plt.savefig("OU.png")

    # Prepare to train network
    num_dims = 10              # Number of OU timesteps
    batch_size = 64            # Batchsize
    num_hidden_units = 12      # GRU cells in the RNN layer

    # Initialize the network and perform a forward pass
    init_func, gru_rnn = stax.serial(Dense(num_hidden_units), Relu,
                                     GRU(num_hidden_units), Dense(1))
    _, params = init_func(key, (batch_size, num_dims, 1))
    
    def mse_loss(params, inputs, targets):
        """ Calculate the Mean Squared Error Prediction Loss. """
        preds = gru_rnn(params, inputs)
        return np.mean((preds - targets)**2)

    @jit
    def update(params, x, y, opt_state):
        """ Perform a forward pass, calculate the MSE & perform a SGD step. """
        loss, grads = value_and_grad(mse_loss)(params, x, y)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, loss
    
    learning_rate = 1e-4
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)
    num_batches = 1000
    train_loss_log = []
    start_time = time.time()
    for batch_idx in range(num_batches):
        x, x_tilde = generate_ou_process(batch_size, num_dims, mu, tau, sigma, noise_std, dt)
        x_in = np.expand_dims(x_tilde[:, :(num_dims-1)], 2)
        y = np.array(x[:, 1:])
        params, opt_state, loss = update(params, x_in, y, opt_state)
        batch_time = time.time() - start_time
        train_loss_log.append(loss)
        if batch_idx % 100 == 0:
            start_time = time.time()
            print("Batch {} | T: {:0.2f} | MSE: {:0.2f} |".format(batch_idx, batch_time, loss))
    return
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--make_plots", action="store_true")
    args = p.parse_args()
    main(make_plots=args.make_plots)
