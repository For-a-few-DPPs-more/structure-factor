#!/usr/bin/env python3
# coding=utf-8

import numpy as np
from mpmath import fp as mpm
from scipy.special import yv, jv


def get_random_number_generator(seed):
    """Turn seed into a np.random.Generator instance
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    raise TypeError(
        "seed must be None, an np.random.Generator or an integer (int, np.integer)")
    return np.random.default_rng(seed)


def roots(N):
    # first N Roots of the Bessel J_(d/2-1) functions divided by pi.
    return np.array([mpm.besseljzero(d/2 - 1, i + 1) for i in range(N)]) / np.pi


def psi(t):
    return t * np.tanh(np.pi * np.sinh(t) / 2)


def get_x(h, zeros):
    return np.pi * psi(h * zeros) / h


def weight(d, zeros):
    return yv(d/2-1, np.pi * zeros) / jv(d/2, np.pi * zeros)


def d_psi(t):
    t = np.array(t, dtype=float)
    d_psi = np.ones_like(t)
    exact_t = t < 6
    t = t[exact_t]
    d_psi[exact_t] = (np.pi * t * np.cosh(t) + np.sinh(np.pi * np.sinh(t))) / (
        1.0 + np.cosh(np.pi * np.sinh(t))
    )
    return d_psi
