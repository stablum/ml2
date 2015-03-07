#!/usr/bin/env python
import numpy as np

from sumproduct import Node, Variable, Factor, bcolors

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
    #f_shape = [] (no need DELME)
    print "vnames",vnames
    for name in vnames:
        v = vs[name]
        fa_vs.append(v) 
        #f_shape.append(v.num_states) (no need DELME)

    fa_name = "_".join(vnames)
    #f_shape = tuple(f_shape) (no need, DELME)
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

def main():
    create_graph()
    """
    print vs
    for name,fa in fs.items():
        print name,fa,fa.f
    print "----"
    for leaf_factor in ['smokes','influenza']:
        fs[leaf_factor].send_sp_msg(vs[leaf_factor])
    for v_name in ['sorethroat','fever','coughing','wheezing']:
        vs[v_name].send_sp_msg(vs[v_name].neighbours[0])
    print "fs.keys()",fs.keys()
    for f_name in ['coughing_bronchitis','wheezing_bronchitis']:
        fs[f_name].send_sp_msg(vs['bronchitis'])
    for f_name in ['sorethroat_influenza','fever_influenza']:
        fs[f_name].send_sp_msg(vs['influenza'])
    for v_name in ['influenza','smokes']:
        vs[v_name].send_sp_msg(fs['bronchitis_influenza_smokes'])
    fs['bronchitis_influenza_smokes'].send_sp_msg(vs['bronchitis'])
    #vs['influenza'].send_sp_msg(fs['bronchitis_influenza_smokes'])
    print 'smokes in_msgs', vs['smokes'].in_msgs
    print 'influenza in_msgs', vs['influenza'].in_msgs
    print 'bronchitis_influenza_smokes in_msgs', fs['bronchitis_influenza_smokes'].in_msgs
    print 'marginal for bronchitis',vs['bronchitis'].marginal()
    """

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
    sum_product(node_list)

def sum_product(node_list):
    node_list_reverse = node_list[:]
    node_list_reverse.reverse()
    for node in node_list + node_list_reverse:
        print bcolors.OKGREEN + str(node) + bcolors.ENDC
        node.send_sp_pending()

if __name__ == "__main__":
    main()

