#!/usr/bin/env python
import numpy as np

from messagepassing import Node, Variable, Factor, bcolors
import launch

vs = {}
fs = {}

"""
    [[[0.9999,0.1],
      [0.0001,0.9]],
     [[0.3,0.01],
      [0.7,0.99]]]
"""

factors_config = [
    (["influenza"],
    [0.95,0.05]
    ),

    (["smokes"],
    [0.8,0.2]
    ),

    (["sorethroat","influenza"],
    [[0.999,0.7],
     [0.001,0.3]]
    ),

    (["fever","influenza"],
    [[0.95,0.1],
     [0.05,0.9]]
    ),

    (["bronchitis","influenza","smokes"],
    [[[0.9999,0.3],
      [0.1,0.01]],
     [[0.0001,0.7],
      [0.9,0.99]]]
    ),

    (["coughing","bronchitis"],
    [[0.93,0.2],
     [0.07,0.8]]
    ),

    (["wheezing","bronchitis"],
    [[0.999, 0.4],
     [0.001, 0.6]]
    ),
]

def create_factor(vnames, values):
    fa_vs = []
    print "vnames",vnames
    for name in vnames:
        v = vs[name]
        fa_vs.append(v) 

    fa_name = "_".join(vnames)
    f = np.array(values)
    fa = Factor(fa_name,f,fa_vs)
    return fa

def create_graph():
    vnames = ["influenza","smokes","sorethroat","fever","bronchitis","coughing","wheezing"]
    for name in vnames:
        vs[name]= Variable(name,2)

    for fc in factors_config:
        fa = create_factor(fc[0],fc[1])
        fs[fa.name] = fa

def create_node_list():
    leaves = []
    for leaf_f in ['smokes','influenza']:
        leaves.append(fs[leaf_f])
    for leaf_v in ['sorethroat','fever','coughing','wheezing']:
        leaves.append(vs[leaf_v])
    
    for le in leaves:
        le.set_pending_except(None)
    
    node_list = leaves[:]
    for f_name in ['coughing_bronchitis','wheezing_bronchitis','sorethroat_influenza','fever_influenza']:
        node_list.append(fs[f_name])
    for v_name in ['influenza','smokes']:
        node_list.append(vs[v_name])
    node_list.append(fs['bronchitis_influenza_smokes'])
    node_list.append(vs['bronchitis'])
    return node_list

def observe_some_variables():
    vs['smokes'].set_observed(0)
    vs['influenza'].set_observed(1)
    vs['bronchitis'].set_observed(1)
    
def main_diagnosis():
    create_graph()
    node_list = create_node_list()
    observe_some_variables()
    launch.sum_product(node_list)
    #launch.max_sum(node_list)
    print fs['bronchitis_influenza_smokes'].marginal_sp()
    #launch.argmax_on_latents(vs, 'ms')

if __name__ == "__main__":
    main_diagnosis()

