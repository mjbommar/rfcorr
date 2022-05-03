# Copyright (c) 2022. Bommarito Consulting, LLC
# SPDX-License-Identifier: Apache-2.0

# package imports
import numpy


def sign_with_zero(x: numpy.array):
    """
    Get the sign of x, but with zero treated as zero.
    Results in three-class/ternary labels.
    :param x:
    :return:
    """
    return numpy.sign(x).astype(int)


def sign_without_zero(x: numpy.array):
    """
    Get the sign of x, but with zero treated as zero.
    Results in two-class/binary labels.
    :param x:
    :return:
    """
    return (x >= 0).astype(int)
