import errno
import os
import math
import signal
import functools
import random
from sympy import sympify
from sympy.functions import Abs

def shrink_a_number(n):
    if n.is_Integer and Abs(n) > 20:
        return n // (10**math.floor(math.log10(abs(n))))
    elif n.is_Float and Abs(n) > 20:
        return n / (10**math.floor(math.log10(abs(n))))
    elif n.is_Rational and Abs(n) > 20:
        return n // (10**math.floor(math.log10(abs(n))))
    return n

# Randomize a number
# if shrink==True, shrink the number to be within 10
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

def random_remove_items(expr, prob_delete=0.1):
    for i in range(10):
        constants = [t for t in expr.atoms() if t not in expr.free_symbols]
        mapping = {}
        for constant in constants:
            if random.random() < prob_delete:
                mapping[constant] = sympify('0')
        expr_removed = expr.subs(mapping)
        if not expr_removed.is_constant():
            return expr_removed
    return expr

def shrink_numbers(expr):
    constants = [t for t in expr.atoms() if t not in expr.free_symbols]
    mapping = {}
    for constant in constants:
        mapping[constant] = shrink_a_number(constant)
    return expr.subs(mapping)

def randomize(expr, prob_change=0.5, shrink=False, remove_items=False):
    if shrink:
        expr = shrink_numbers(expr)
    constants = [t for t in expr.atoms() if t not in expr.free_symbols]
    mapping = {}
    for constant in constants:
        if random.random() < prob_change:
            mapping[constant] = randomize_number(constant)
    expr_subed = expr.subs(mapping)
    if remove_items:
        return random_remove_items(expr_subed)
    else:
        return expr_subed

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
