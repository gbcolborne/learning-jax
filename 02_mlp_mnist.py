""" Trains an MLP on MNIST. """
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad, random
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
import torch

key = random.PRNGKey(1)
