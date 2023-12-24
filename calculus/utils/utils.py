import errno
import os
import math
import signal
import functools
import random
from sympy import sympify
from sympy.functions import Abs
from sympy.core.rules import Transform

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

def round_all_floats(f, num_digits=2):
    return f.xreplace(Transform(lambda x: x.round(num_digits), lambda x: isinstance(x, sp.Float)))

def get_all_constants(expr):
    return [t for t in expr.atoms() if t not in expr.free_symbols]

def get_coefficients_and_exponents(f):
    variables = list(f.free_symbols)
    assert len(variables)<=1, "Expression having multiple variable " + str(f)
    if len(variables) == 0:
        return list()
    t = variables[0]
    return [[float(x) for x in term.as_coeff_exponent(t)] for term in f.as_ordered_terms()]

# Return coeffs of a polynomial from lowest power to highest
def get_polynomial_coeffs(f, max_power=6):
    result = [0.0]*(max_power+1)
    coeff_pairs = get_coefficients_and_exponents(f)
    for p in coeff_pairs:
        coeff, exp = float(p[0]), float(p[1])
        idx = int(round(exp))
        if idx < 0 or idx > max_power:
            continue
        result[idx] = coeff
    return result

# Only keep the polynomial part
@timeout(5)
def filter_non_polynomial(f):
    variables = list(f.free_symbols)
    assert len(variables)<=1, "Expression having multiple variable " + str(f)
    if len(variables) == 0:
        return sympify(0)
    t = variables[0]
    terms_kept = []
    terms = f.as_ordered_terms()
    for term in terms:
        coeff, exp = term.as_coeff_exponent(t)
        if coeff.is_constant():
            terms_kept.append(term)
    if len(terms_kept) == 0:
        return sympify(0)
    return sum(terms_kept)

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

