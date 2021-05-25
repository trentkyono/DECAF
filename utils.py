import numpy as np
import networkx as nx
import pandas as pd

# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(G, base_mean = 0, base_var = 0.3, mean = 0, var = 1, SIZE = 10000, err_type = 'normal', perturb = [], sigmoid = True, expon = 1.1):
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean,var,SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == 'gumbel':
                g.append(np.random.gumbel(base_mean, base_var,SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var,SIZE))
            


    for o in order:
        for edge in list_edges:
            if o == edge[1]: # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1/1+np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g,0,1)

    return pd.DataFrame(g, columns = list(map(str, list_vertex)))

