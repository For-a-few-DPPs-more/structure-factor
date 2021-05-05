#!/usr/bin/env python3
# coding=utf-8

import numpy as np


def get_random_number_generator(seed):
    """Turn seed into a np.random.Generator instance
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    raise TypeError(
        "seed must be None, an np.random.Generator or an integer (int, np.integer)")
