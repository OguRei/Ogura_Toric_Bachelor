from itertools import combinations
import numpy as np
import networkx as nx
import time
from typing import List, Tuple
from tqdm import trange
from param import param
from toric_code import ToricCode

def decode(coordinate, param):
    g = nx.Graph()
    for u, v in combinations(coordinate, 2):
        x_distance = min(np.abs(u[0]-v[0]),param.code_distance-np.abs(u[0]-v[0]))
        y_distance = min(np.abs(u[1]-v[1]),param.code_distance-np.abs(u[1]-v[1]))
        weight = -(x_distance * 100 + y_distance * (1+100))
        g.add_edge(tuple(u), tuple(v), weight = weight)
    matching = nx.algorithms.max_weight_matching(g,maxcardinality=True)
    return matching

def evaluate(n_iter):
    count = 0
    param_eva = param(p=0.05, size=9)
    toric_code=ToricCode(param_eva)
    spend=0
    for _ in trange(n_iter):
        errors= toric_code.generate_errors()
        #print(errors)
        syndromeX = toric_code.generate_syndrome_X(errors)
        #print(syndromeX)
        syndromeZ = toric_code.generate_syndrome_Z(errors)
        #print(syndromeZ)
        coordinate_X = list(zip(*np.where(syndromeX==1)))
        #print(coordinate_X)
        coordinate_Z = list(zip(*np.where(syndromeZ==1)))
        #print(coordinate_Z)
        before = time.perf_counter()
        matching_X = decode(coordinate_X, param_eva)
        #print(matching_X)
        matching_Z = decode(coordinate_Z, param_eva)
        #print(matching_Z)
        spend += time.perf_counter()-before
        for u,v in matching_Z:
            errors = toric_code.decode_X_error(errors,u,v)
        for u,v in matching_X:
            errors = toric_code.decode_Z_error(errors,u,v)
        #print(errors)
        if np.all(toric_code.generate_syndrome_X(errors)==0) and np.all(toric_code.generate_syndrome_Z(errors)==0):
            if toric_code.not_has_non_trivial_X(errors) and toric_code.not_has_non_trivial_Z(errors):
                count += 1
                #print("True")
    print(str(spend/n_iter) + str(" seconds"))
    print(f"logical error rates: {n_iter-count}/{n_iter}", (n_iter-count)/n_iter)

evaluate(10000)





