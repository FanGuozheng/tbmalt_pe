#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:05:18 2022

@author: gz_fan
"""
import logging


def get_logger(name=__name__, level=logging.INFO):
    """A template for logging functions."""
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
