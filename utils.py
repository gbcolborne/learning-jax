import time, functools
import jax.numpy as np


def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x for k classes """
    return np.array(x[:, None] == np.arange(k), dtype)


def timer(func):
    """Decorator for timing function execution. From
    `https://realpython.com/python-timer/#a-python-timer-decorator`. """

    # First, decorate the inner function as a wrapper, so that any
    # function that is decorated by our timer will keep its name.
    # Then, define the inner function that wraps `func` such that we
    # time its execution.
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed = toc - tic
        print(f"Elapsed time: {elapsed:0.6f} seconds.")
        return value
    return wrapper_timer


def timed_exec(func, *args, **kwargs):
    """ Timed execution of a function. """
    tic = time.perf_counter()
    value = func(*args, **kwargs)
    toc = time.perf_counter()
    print(f"Elapsed time: {elapsed:0.6f} seconds.")
    return value
