from pyspark import SparkContext
import json
import time
import csv
from collections import Counter
from itertools import combinations
from math import factorial, ceil
import random

def Girvan_Newman(x,nodes,neighbors_map):
    root = x
    nodes_level = {node:float('inf') for node in nodes}
    nodes_label = {node:0 for node in nodes}
    nodes_credit = {node:1 for node in nodes}

    nodes_level[root] = 0
    nodes_label[root] = 1
    queue = [root]
    visited = []
    # Label
    while queue:
        cur_node = queue.pop(0)
        visited.append(cur_node)
        cur_node_level = nodes_level[cur_node]
        cur_node_label = nodes_label[cur_node]
        for nb_node in neighbors_map[cur_node]:
            if nodes_level[nb_node]>cur_node_level:
                nodes_level[nb_node] = cur_node_level+1
                nodes_label[nb_node] += cur_node_label
                if nb_node not in queue:
                    queue.append(nb_node)
    
    #Credit
    result = []
    while visited:
        cur_node = visited.pop()
        cur_node_level = nodes_level[cur_node]
        cur_node_label = nodes_label[cur_node]
        cur_node_credit = nodes_credit[cur_node]
        for nb_node in neighbors_map[cur_node]:
            if nodes_level[nb_node]<cur_node_level:
                edge = tuple(sorted([cur_node,nb_node]))
                edge_credit = nodes_label[nb_node]*cur_node_credit/cur_node_label
                nodes_credit[nb_node] += edge_credit
                result.append((edge,edge_credit))

    return result
                    
def csv_to_rdd(path):
    return sc.textFile(path)\
            .zipWithIndex().filter(lambda x: x[1]>0)\
            .map(lambda x:tuple(x[0].split(",")))

def calc_modularity(cmnts,neighbors_map):
    m = sum([len(v) for v in neighbors_map.values()])/2
    result = 0
    def Aij(i,j):
        return int(j in neighbors_map[i])
    def k(i):
        return len(neighbors_map[i])
    for cmnt_nodes in cmnts:
        for i,j in combinations(cmnt_nodes,2):
            result += Aij(i,j)-(k(i)*k(j)/(2*m))
    result = result/(2*m)
    return result
        
def remove_edge(graph,edge):
    i,j = edge
    graph[i].remove(j)
    graph[j].remove(i)
    return graph


def find_communities(nodes,neighbors_map):
    
    def find_cmnt(root):
        queue = []
        visited = []
        queue.append(root)
        visited.append(root)
        while queue:
            node = queue.pop(0)
            for nb_node in neighbors_map[node]:
                if nb_node not in visited:
                    visited.append(nb_node)
                    queue.append(nb_node)
        return sorted(visited)
    result = []
    remaining_nodes = set(nodes)
    while remaining_nodes:
        random_node = random.choice(list(remaining_nodes))
        found_cmnt = find_cmnt(random_node)
        result.append(found_cmnt)
        remaining_nodes = remaining_nodes-set(found_cmnt)
    return result

def main(input_path, btwness_output_path, cmnt_output_path):
    start_time = time.time()
    
    raw_rdd = csv_to_rdd(input_path).map(lambda x: (x[1],x[0])).partitionBy(NUM_PART,lambda x: (23*hash(x)+109)%NUM_PART)
    corated_userset = raw_rdd.groupByKey().mapValues(set)
    edges_rdd = corated_userset.flatMap(lambda x: [(tuple(sorted(list(pair))),1) for pair in combinations(x[1],2)])\
                        .reduceByKey(lambda a,b: a+b)\
                        .filter(lambda x: x[1]>=THRES)\
                        .map(lambda x: x[0]).flatMap(lambda x: [x,(x[1],x[0])])
    neighbors_map = edges_rdd.groupByKey().mapValues(list).collectAsMap()
    nodes = list(neighbors_map.keys())
    nodes_rdd = sc.parallelize(nodes)
    # Calculate Betweeness and Output
    edge_btwness = nodes_rdd.flatMap(lambda x: Girvan_Newman(x,nodes,neighbors_map))\
                    .reduceByKey(lambda a,b: a+b)\
                    .map(lambda x: (x[0],x[1]/2))\
                    .sortBy(lambda x: [-x[1],x[0][0]])\
                    .collect()
    with open(btwness_output_path,'w') as file:
        for edge,score in edge_btwness:
            file.write(f"{str(edge)},{round(score,5)}")
            file.write("\n")
    
    # Detect Communities
    past_cmnt = find_communities(nodes,neighbors_map)
    ## Initial Modularity
    past_modularity = calc_modularity(past_cmnt,neighbors_map)
    while True:
        edge2remove = edge_btwness[0][0]
        ## Remove edge
        neighbors_map = remove_edge(neighbors_map,edge2remove)
        ## Detect Communities
        current_cmnt = find_communities(nodes,neighbors_map)
        ## Re-calculate Modularity
        current_modularity = calc_modularity(current_cmnt,neighbors_map)
        ## Compare Modularity
        if current_modularity<past_modularity:
            break
        ## Re-compute btwness
        edge_btwness = nodes_rdd.flatMap(lambda x: Girvan_Newman(x,nodes,neighbors_map))\
                    .reduceByKey(lambda a,b: a+b)\
                    .map(lambda x: (x[0],x[1]/2))\
                    .sortBy(lambda x: [-x[1],x[0][0]])\
                    .collect()
        past_modularity = current_modularity
        past_cmnt = current_cmnt
    cmnt_result = sorted(past_cmnt, key=lambda x: [len(x),str(x)])
    with open(cmnt_output_path,'w') as file:
        for comm in cmnt_result:
            file.write(str(comm)[1:-1]+"\n")
    print(time.time()-start_time)

sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    NUM_PART = 30
    import sys
    THRES = int(sys.argv[1])
    main(sys.argv[2], sys.argv[3], sys.argv[4])