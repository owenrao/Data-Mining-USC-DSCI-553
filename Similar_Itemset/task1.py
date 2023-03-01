from pyspark import SparkContext
import json
import time
import csv
from collections import Counter
from itertools import combinations
from math import factorial, ceil
import random

def gen_prime_num(target):
    result = []
    count = 0
    i = 2
    while count < target:
        for j in range(2,i):
            if (i%j) == 0:
                break
        else:
            result.append(i)
            count += 1
        i += 1
    return result

def gen_hash_func(m):
    a = random.randint(100,999)
    b = random.randint(0,99)
    p = random.choice(prime_set)
    return lambda x: ((a*x+b)%p)%m

def split_bands(l, b):
    biz_id,x = l
    k, m = divmod(len(x), b)
    return (((i,tuple(x[i*k+min(i, m):(i+1)*k+min(i+1, m)])),biz_id) for i in range(b))

def gen_b_r(N):
    target = s*0.65 #**N
    epsilon = 0.001
    f = lambda x: 1/(x**(x/N))
    def d_n_c(start, end):
        if (end-start)<=1:
            if abs(f(start)-target) <= abs(f(end)-target):
                return start
            else: return end

        x = (end+start)/2
        fx = f(x)
        if abs(fx-target)<=epsilon:
            return x
        
        if fx<target:
            return d_n_c(start, x)

        else:
            return d_n_c(x+1, end)
    b = int(round(d_n_c(1, N),0))
    r = int(round(N/b,0))
    return b,r

def gen_input_mat(part,item_list):
    basket = [(tup[0],tup[1]) for tup in part]
    result = []
    for i in basket:
        result.append((i[0],[1 if item in i[1] else 0 for item in item_list]))
    return result

def gen_sig_mat(x,hash_funcs):
    biz_id,items = x
    #items = [i for i in x]
    result = [float('inf') for i in range(NUM_HASH)]
    for i,item in enumerate(items):
        if item==1:
            for k in range(NUM_HASH):
                hk = hash_funcs[k](i)
                if hk<result[k]:
                    result[k] = hk
    return (biz_id,result)

def jaccard_sim(a,b):
    return len(a.intersection(b))/len(a.union(b))

def map_biz_id(x,id_biz_map_dict):
    pair = tuple(sorted([id_biz_map_dict[x[0][0]],id_biz_map_dict[x[0][1]]]))
    return (pair,x[1])

def main(input_path, output_path):
    result = {}
    start_time = time.time()
    raw_rdd = sc.textFile(input_path)
    rdd = raw_rdd.zipWithIndex().filter(lambda x: x[1]>0)
    #size = rdd.top(1,key=lambda x: x[1])[0][-1]
    #num_partition = size//20000+1
    rdd = rdd.map(lambda x:tuple(x[0].split(",")))
    user_list = rdd.map(lambda x:x[0]).distinct().sortBy(lambda x: x).collect()
    biz_id_map = rdd.map(lambda x:x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collect()
    biz_id_map_dict = {biz_id:i for biz_id,i in biz_id_map}
    id_biz_map_dict = {i:biz_id for biz_id,i in biz_id_map}
    N = len(user_list)
    print(N)
    b,r = gen_b_r(NUM_HASH)
    #b = 30
    #r = N/30
    print(b,r)
    rdd = rdd.map(lambda x: (biz_id_map_dict[x[1]],x[0])).partitionBy(b, lambda x: (109*hash(x)+23)%b).groupByKey().mapValues(set).persist()
    input_mat = rdd.collectAsMap()
    input_mat_rdd = rdd.mapPartitions(lambda x: gen_input_mat(x,user_list))
    hash_funcs = [gen_hash_func(N*2) for i in range(NUM_HASH)]
    sig_mat_rdd = input_mat_rdd.map(lambda x: gen_sig_mat(x,hash_funcs))
    candidate_rdd = sig_mat_rdd.flatMap(lambda x: split_bands(x, b)).partitionBy(b, lambda x: x[0]).groupByKey().mapValues(list).filter(lambda x: len(x[1])>1).flatMap(lambda x: [pair for pair in combinations(x[1],2)]).distinct()#.sortBy(lambda x: [x[0],x[1]]).collect()
    candidate_pairs = candidate_rdd.map(lambda x: (x, jaccard_sim(input_mat[x[0]],input_mat[x[1]]))).filter(lambda x: x[1]>=s).map(lambda x: map_biz_id(x,id_biz_map_dict)).sortBy(lambda x: [x[0],x[1]]).collect()
    #for i,pair in enumerate(candidate_pairs):
        #pass
    
    #print(candidate_pairs.take(1))
    #print(input_mat_rdd.take(1))
    #print(sig_mat_rdd.take(1))
    
    with open(output_path,'w') as file:
        writer = csv.writer(file)
        writer.writerow(("business_id_1","business_id_2", "similarity"))
        for pair, sim in candidate_pairs:
            writer.writerow((pair[0], pair[1], sim))
   
    print(time.time()-start_time)

sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
s = 0.5
if __name__ == "__main__":
    NUM_HASH = 100
    #NUM_BIN = 10
    prime_set = gen_prime_num(NUM_HASH*5)
    import sys
    main(sys.argv[1], sys.argv[2])