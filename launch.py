#!/usr/bin/env python

import numpy as np

from messagepassing import Node, Variable, Factor, bcolors

def run_on_list(node_list,algo='sp'):
    node_list_reverse = node_list[:]
    node_list_reverse.reverse()
    for node in node_list + node_list_reverse:
        print bcolors.OKGREEN + str(node) + bcolors.ENDC
        node.send_pending(algo)

def sum_product(node_list):
    run_on_list(node_list,algo='sp')

def max_sum(node_list):
    run_on_list(node_list,algo='ms')

def argmax_on_latents(_vs, algo):
    for v in latent_variables(_vs):
        print v.name,'=', v.argmax(algo)

def latent_variables(_vs):
    return [ curr for curr in _vs.values() if curr.is_latent() ]

def factors_with_latents(_fs):
    return [ curr for curr in _fs.values() if curr.has_latents() ]

