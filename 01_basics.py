"""Learning the basics of jax using the tutorial at
https://roberttlange.github.io/posts/2020/03/blog-post-10/.

"""
import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, random
from utils import timer

# Generate key to generate random numbers
key = random.PRNGKey(1)

# Generate a random matrix
x = random.uniform(key, (1000, 1000))

# Compare running times of 3 different matrix multiplications
@timer
def dot_numpy(x, y):
    return onp.dot(x, x)

@timer
def dot_jax(x, y):
   return np.dot(x, x)

@timer
def dot_jax_block(x, y):
    return np.dot(x, x).block_until_ready()

print("Testing dot with numpy")
dot_numpy(x, x)
print("Testing dot with jax (without block_until_ready)")
dot_jax(x, x)
print("Testing dot with jax")
dot_jax_block(x, x)

# Test jit
def relu(x):
    return np.maximum(0,x)

@timer
def timed_relu(x):
    return relu(x)

@timer
@jit
def timed_jit_relu(x):
    return np.maximum(0,x)

print("Testing relu without jit")
timed_relu(x).block_until_ready()
print("Testing relu with jit")
timed_jit_relu(x).block_until_ready()
print("Testing relu with jit a second time")
timed_jit_relu(x).block_until_ready()

# Test grad
def FiniteDiffGrad(x):
    """ Compute the finite difference derivative approx for the ReLU"""
    return np.array((relu(x + 1e-3) - relu(x - 1e-3)) / (2 * 1e-3))

# Compare the Jax gradient with a finite difference approximation
print("Computing gradients...")
d1 = jit(grad(jit(relu)))(2.)
d2 = FiniteDiffGrad(2.)
print("Jax Grad: ", d1)
print("Approximate Gradient:", d2)

# Test vmap
batch_size = 32
dim_in = 100
dim_out = 512
X = random.normal(key, (batch_size, dim_in)) # single batch of vectors
params = [random.normal(key, (dim_out, dim_in)),
          random.normal(key, (dim_out, ))]

def relu_layer(params, x):
    """ ReLU for single sample. """
    return relu(np.dot(params[0],x) + params[1])

def batched_relu_layer(params, x):
    """ Error prone batch version. """
    return relu(np.dot(X, params[0].T) + params[1])

def vmap_relu_layer(params, x):
    """ Batched version using vmap. """
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))

out1 = np.stack([relu_layer(params, X[i, :]) for i in range(X.shape[0])])
out2 = batched_relu_layer(params, X)
out3 = vmap_relu_layer(params, X)


