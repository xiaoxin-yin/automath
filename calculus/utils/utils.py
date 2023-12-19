import errno
import os
import signal
import functools
import random
from sympy import sympify

def randomize_number(n):
    if n.is_Integer:
        for i in range(10):
            m = n + random.randint(-2,2)
            if (n > 0 and m > 0) or (n < 0 and m < 0):
                return m
        return n
    elif n.is_Float:
        m = n * random.choice([sympify('1/2'), 2, 3])
        return m
    return n

def randomize(expr, prob_change=0.5):
    constants = [t for t in expr.atoms() if t not in expr.free_symbols]
    mapping = {}
    for constant in constants:
        if random.random() < prob_change:
            mapping[constant] = randomize_number(constant)
    return expr.subs(mapping)

class TimeoutError(Exception):
    pass

def timeout(seconds=15, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator
