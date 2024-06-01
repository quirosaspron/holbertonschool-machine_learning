#!/usr/bin/env python3
"""calculates the cost with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Returns: Tensor with the L2 cost for each layer"""
    l2_losses = model.losses
    return cost + l2_losses
