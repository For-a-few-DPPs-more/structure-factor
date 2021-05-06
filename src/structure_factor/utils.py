#!/usr/bin/env python3
# coding=utf-8

import numpy as np


def get_random_number_generator(seed):
    return np.random.default_rng(seed)
