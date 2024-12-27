#!/usr/bin/env python3

"""
Utilities class for tree building and analysis.

Author: Tucker Hwang (tucker_hwang@berkeley.edu)
"""

import logging

import numpy as np
from functools import reduce
import itertools

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s')
handler.setFormatter(formatter)

def linbins(xmin, xmax, nbins):
    return np.linspace(xmin, xmax, nbins+1)

def logbins(xmin, xmax, nbins):
    return np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)

def find_key_path(rules, obs, path = []):
    for key, value in rules.items():
        current_path = path + [key]
        if key == obs:
            return current_path
        elif isinstance(value, dict):
            try:
                return find_key_path(value, obs, current_path)
            except InvalidTargetError:
                continue
    raise InvalidTargetError("No rule for target found.")

def get_by_path(rules, keypath):
    return reduce(lambda d, key: d[key], keypath, rules)

def search_backwards(rules, keypath):
    # Start with full rules dict and empty path
    subdict = rules
    current_path = []
    
    # First yield the value at the full keypath
    for key in keypath:
        subdict = subdict[key]
        current_path.append(key)
    yield subdict

    # Then yield dictionaries for each parent path
    for i in range(len(keypath)-1, 0, -1):
        subdict = rules
        for key in keypath[:i]:
            subdict = subdict[key]
        yield subdict

    # Finally yield the original rules dict
    yield rules

def search_forwards(rules, keypath):
    # Start with the original rules dict
    subdict = rules
    yield subdict
    
    # Yield dictionaries for each subpath
    for i in range(1, len(keypath)):
        subdict = rules  # Reset to original rules dict
        for key in keypath[:i]:   # Navigate to current level
            subdict = subdict[key]
        yield subdict
    
    # Finally yield the value at the full keypath
    subdict = rules     # Reset to original rules
    for key in keypath:          # Navigate through full path
        subdict = subdict[key]
    yield subdict

def build_recipe(rules, keypath):
    recipe = []
    for subdict in search_forwards(rules, keypath):
        if 'dir' in subdict.keys():
            for directive in subdict['dir']:
                recipe.append(directive)
    if recipe[-1][0] != 'h':
        raise InvalidTargetError("Last directive not a histogram.")
    return recipe

def find_parameters(rules, keypath):
    params = {}
    for subdict in search_forwards(rules, keypath):
        if 'params' in subdict.keys():
            for name, vals in subdict['params'].items():
                if name in params.keys():
                    logger.warning("Found a duplicate parameter defined, skipping.")
                else:
                    params[name] = vals
    return params

def find_relevant_parameters(params, recipe):
    paramkeys = params.keys()
    relparams = {param: False for param in paramkeys}
    for directive in recipe:
        for relparam in paramkeys:
            if relparam in str(directive[-1]):
                relparams[relparam] = True
    return {param: vals for param, vals in params.items() if relparams[param]}

# find source tree
def find_source_tree(config, keypath):
    for subdict in search_backwards(config, keypath):
        if 'tree' in subdict.keys():
            return subdict['tree']
    raise InvalidTargetError("No source tree found for target.")

def parse_directives(relparams, recipe):
    parsed_directives = {}
    names = relparams.keys()
    for vals in itertools.product(*relparams.values()):
        reference = {name: val for name, val in zip(names, vals)}
        name = '_'.join([f'{name}_{val}' for name, val in reference.items()])
        # print(reference)
        # given the reference dict, replace all instances of each param with its value
        parsed_recipe = [] # list of directives, with vars replaced
        for directive in recipe:
            detail = directive[-1] # obtain actual directive
            for parname, val in reference.items(): # replace vars
                detail = detail.replace(parname, f"{val}") if isinstance(detail, str) else detail
            parsed_directive = list(directive)
            parsed_directive[-1] = detail
            parsed_recipe.append(parsed_directive)
        parsed_directives[name] = parsed_recipe
    return parsed_directives

def calc_bins(params):
    bintype, binmin, binmax, nbins = params
    if bintype in ['log', 'lg']:
        return logbins(binmin, binmax, nbins)
    elif bintype in ['lin', 'ln']:
        return linbins(binmin, binmax, nbins)
    else:
        raise InvalidTargetError(f"Binning type {bintype} invalid.")

class InvalidTargetError(Exception):
    def __init__(self, msg):
        self.msg = msg

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'DEBUG': '\033[34m',
        'INFO': '\033[32m',
        'CRITICAL': '\033[35m'
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        if color:
            # Color the entire line
            formatted_msg = super().format(record)
            return f"{color}{formatted_msg}{self.RESET}"
        return super().format(record)