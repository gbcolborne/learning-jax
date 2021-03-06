""" Trains an MLP on MNIST. Source: `https://roberttlange.github.io/posts/2020/03/blog-post-10/`. """
import time
import jax.numpy as np
from jax import jit, vmap, value_and_grad, random
from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers
import torch
from torchvision import datasets, transforms
from utils import one_hot

def initialize_mlp(sizes, key):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, len(sizes))
    # Initialize a single layer with Gaussian weights -  helper function
    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(params, x):
    """ ReLu layer for single sample """
    z = np.dot(params[0], x) + params[1]
    a = np.maximum(0, z)
    return a


def mlp_forward_pass(params, in_array):
    """ Forward pass for a single sample. """
    activations = in_array

    # Loop over the ReLU hidden layers
    for w, b in params[:-1]:
        activations = relu([w, b], activations)

    # Compute output logits
    final_w, final_b = params[-1]
    logits = np.dot(final_w, activations) + final_b
    logsoftmax = logits - logsumexp(logits) 
    return logsoftmax


def ce_loss(params, in_arrays, targets):
    """Cross-entropy loss. Note: `batched_forward` must be a globally
    defined function. This is because a @jit decorated function's
    "arguments and return value should be arrays, scalars, or (nested)
    standard Python containers (tuple/list/dict) thereof"
    (https://github.com/google/jax/blob/0e92124c5b0c9b2ab4ed306dc4491413062fd3db/jax/api.py#L120).

    """
    preds = batched_forward(params, in_arrays) 
    return -np.sum(preds * targets)


# Only the update function needs jit, as it encapsulates the other
# functions, e.g. relu, forward pass, loss, etc.
@jit
def update(params, x, y, opt_state):
    """Compute the gradient for a batch and update the parameters. Note:
    `batched_forward`, `get_params`, and `opt_update` must be globally
    defined functions. This is because a @jit decorated function's
    "arguments and return value should be arrays, scalars, or (nested)
    standard Python containers (tuple/list/dict) thereof"
    (https://github.com/google/jax/blob/0e92124c5b0c9b2ab4ed306dc4491413062fd3db/jax/api.py#L120).

    """
    value, grads = value_and_grad(ce_loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def accuracy(params, data_loader, num_classes):
    """Compute the accuracy of an MLP for a given dataloader. Note:
    `batched_forward` must be a globally defined function. This
    is because a @jit decorated function's "arguments and return value
    should be arrays, scalars, or (nested) standard Python containers
    (tuple/list/dict) thereof"
    (https://github.com/google/jax/blob/0e92124c5b0c9b2ab4ed306dc4491413062fd3db/jax/api.py#L120).

    """
    acc_total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        images = np.array(data)
        # Flatten
        images = images.reshape(data.size(0), 28*28)
        targets = one_hot(np.array(target), num_classes)
        target_class = np.argmax(targets, axis=1)
        logits = batched_forward(params, images)
        predicted_class = np.argmax(logits, axis=1)
        acc_total += np.sum(predicted_class == target_class)
    return acc_total/len(data_loader.dataset)


def run_mnist_training_loop(train_loader, test_loader, num_epochs, opt_state):
    """Training loop over for MNIST.  Note: `batched_forward`,
    `opt_update` and `get_params` must be globally defined
    functions. This is because a @jit decorated function's "arguments
    and return value should be arrays, scalars, or (nested) standard
    Python containers (tuple/list/dict) thereof"
    (https://github.com/google/jax/blob/0e92124c5b0c9b2ab4ed306dc4491413062fd3db/jax/api.py#L120).

    """
    num_classes = 10

    # Lists for logging
    log_acc_train, log_acc_test, train_loss = [], [], []

    # Get the initial set of parameters
    params = get_params(opt_state)

    # Get initial accuracy after random init
    train_acc = accuracy(params, train_loader, num_classes)
    test_acc = accuracy(params, test_loader, num_classes)
    log_acc_train.append(train_acc)
    log_acc_test.append(test_acc)
    print("Epoch 0 | T: 0.00 | Train A: {:0.3f} | Test A: {:0.3f}".format(train_acc, test_acc))

    # Loop over the training epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Flatten the image into a vector for the MLP
            x = np.array(data).reshape(data.size(0), 28*28)
            y = one_hot(np.array(target), num_classes)
            params, opt_state, loss = update(params, x, y, opt_state)
            train_loss.append(loss)

        epoch_time = time.time() - start_time
        train_acc = accuracy(params, train_loader, num_classes)
        test_acc = accuracy(params, test_loader, num_classes)
        log_acc_train.append(train_acc)
        log_acc_test.append(test_acc)
        print("Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f}".format(epoch+1, epoch_time,
                                                                                  train_acc, test_acc))
    return train_loss, log_acc_train, log_acc_test


def train_mlp(key, batch_size, train_loader, test_loader):
    """Train an MLP using data loaders."""
    
    # Get list of tuples of (w,b) layer weights
    layer_sizes = [784, 512, 512, 10]
    params = initialize_mlp(layer_sizes, key) 

    # Make a batched version of the forward pass
    global batched_forward
    batched_forward = vmap(mlp_forward_pass, in_axes=(None, 0), out_axes=0)

    # Define an optimizer 
    learning_rate = 1e-3
    global get_params
    global opt_update
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)
    num_epochs = 10

    # Run training loop
    train_loss, train_log, test_log = run_mnist_training_loop(train_loader,
                                                              test_loader,
                                                              num_epochs,
                                                              opt_state)
    return None

if __name__ == "__main__":
    # Generate key to generate random numbers
    key = random.PRNGKey(1)
    
    # Get data
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,),(0.3081,))])),
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,),(0.3081,))])),
        batch_size=batch_size,
        shuffle=True)

    # Train model
    train_mlp(key, batch_size, train_loader, test_loader)







