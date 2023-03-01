from pyspark import SparkContext
import json
import time
from collections import Counter
from itertools import combinations
from math import factorial, ceil
import csv
import random

cb = lambda n,k: factorial(n)/(factorial(k)*factorial(n - k))

def case_1_import(x):
    x_ = x[0].split(",")
    return (x_[0],x_[1])

def case_2_import(x):
    x_ = x[0].split(",")
    return (x_[1],x_[0])

def gen_bit_maps(baskets, bucket_num, hash_1, n):
    bit_map_1 = {i:0 for i in range(bucket_num)}
    for basket in baskets:
        combs = combinations(basket,n)
        for itset in combs:
            bit_map_1[hash_1(itset)] += 1
    bit_map_1 = {key:1 if value>= local_thres else 0 for key,value in bit_map_1.items()}
    return bit_map_1

def multi_hash(part):
    freq_itsets = []
    baskets = [tup[1] for tup in part]
    N = max([len(x) for x in baskets])
    L = len(baskets)
    item_counter = Counter([ele for tup in baskets for ele in tup])
    prev_freq_itsets = [{ele} for ele,value in item_counter.items() if value>=local_thres]
    freq_itsets += prev_freq_itsets
    bucket_num = int(1/4*L*cb(len(prev_freq_itsets),2)/local_thres)+2
    hash_1 = lambda x: hash(tuple(sorted(list(x))))%bucket_num
    bit_map_1 = gen_bit_maps(baskets, bucket_num, hash_1, 2)
    skip = 0
    for n in range(2,N+1):
        #print("Constructing")
        # Construct
        item_counter = []
        for i,j in combinations(prev_freq_itsets,2):
            if len(i&j)==(n-2):
                temp = i.union(j)
                row = [temp,0]
                if row not in item_counter and skip+bit_map_1[hash_1(temp)]:
                    item_counter.append(row)
        # Count
        #print("Counting")
        for row in item_counter:
            itemset = row[0]
            for basket in baskets:
                if itemset.issubset(basket):
                    row[1] += 1
        prev_freq_itsets = [row[0] for row in item_counter if row[1]>=local_thres]
        freq_itsets += prev_freq_itsets
        if prev_freq_itsets == []:
            break
        # BitMap
        #print("bitmapping")
        possible_comb_count = cb(len(prev_freq_itsets),2)
        if possible_comb_count>1000:
            bucket_num = int(3/4*possible_comb_count/local_thres)+2
            bit_map_1= gen_bit_maps(baskets, bucket_num, hash_1, n+1)
            skip = 0
        else: skip = 1
    return [tuple(sorted(list(i))) for i in freq_itsets]

def apriori(part):
    freq_itsets = []
    baskets = [tup[1] for tup in part]
    p = local_thres
    N = max([len(x) for x in baskets])
    item_counter = Counter([ele for tup in baskets for ele in tup])
    prev_freq_itsets = [{ele} for ele,value in item_counter.items() if value>=p]
    freq_itsets += prev_freq_itsets
    for n in range(2,N+1):
        print("n:",n)
        # Construct
        item_counter = []
        for i,j in combinations(prev_freq_itsets,2):
            if len(i&j)==(n-2):
                row = [i.union(j),0]
                if row not in item_counter:
                    item_counter.append(row)
        # Count
        for row in item_counter:
            itemset = row[0]
            for basket in baskets:
                if itemset.issubset(basket):
                    row[1] += 1
        prev_freq_itsets = [row[0] for row in item_counter if row[1]>=p]
        freq_itsets += prev_freq_itsets
        if prev_freq_itsets == []:
            break
    return [tuple(sorted(list(i))) for i in freq_itsets]
    
def count_frequency(part,candidate_list):
    baskets = [tup[1] for tup in part]
    counter = {cand:0 for cand in candidate_list}
    for basket in baskets:
        for cand in candidate_list:
            if set(cand).issubset(basket):
                counter[cand] += 1
    return [(cand,value) for cand,value in counter.items()]
    
def printf(l):
    result = ""
    n = 1
    while True:
        temp = [str(i) for i in l if len(i)==n]
        if len(temp) == 0:
            break
        result += (",".join(temp)) + 2*"\n"
        n += 1
    result = result.replace(",)",")")
    return result
    
def SON(rdd):
    # Pass 1
    candidate_list = rdd.mapPartitions(multi_hash).distinct().sortBy(lambda x: [len(x),str(x)]).collect()
    # Pass 2
    frequent_itemsets = rdd.mapPartitions(lambda x: count_frequency(x,candidate_list)).reduceByKey(lambda a,b: a+b).filter(lambda x: x[1]>=thres).map(lambda x: x[0]).sortBy(lambda x: [len(x),str(x)]).collect() #
    #print(frequent_itemsets)
    return candidate_list, frequent_itemsets
    
def preprocess(x):
    row = x[0].split(",")
    date = str(row[0]).strip('"')
    cus_id = str(row[1]).strip('"')
    prod_id = str(row[5]).strip('"')
    #prod_id = (13-len(prod_id))*"0"+prod_id
    return (date+"-"+cus_id, prod_id)
    

def main(filter_thres, input_path, output_path):
    filter_thres = int(filter_thres)
    result = {}
    start_time = time.time()
    raw_rdd = sc.textFile(input_path)
    raw_rdd = raw_rdd.zipWithIndex().filter(lambda x: x[1]>0)
    num_partition = raw_rdd.getNumPartitions()
    rdd = raw_rdd.map(preprocess).partitionBy(num_partition, lambda x: hash(x))
    intermediate_content = rdd.collect()
    with open("customer_product.csv","w", encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(("DATE-CUSTOMER_ID", "PRODUCT_ID"))
        for r1, r2 in intermediate_content:
            writer.writerow((r1, r2))
    rdd = rdd.groupByKey().mapValues(set).filter(lambda x: len(x[1])>filter_thres)

    global local_thres
    local_thres = thres/num_partition
    candidate_list, frequent_itemsets = SON(rdd)
    with open(output_path,"w") as file:
        file.write("Candidates: \n") 
        file.write(printf(candidate_list)) 
        file.write("Frequent Itemsets: \n") 
        file.write(printf(frequent_itemsets)) 
    duration = time.time()-start_time
    print(f"Duration:{duration}")
    return f"Duration:{duration}"

sc = SparkContext('local[*]', 'task1').getOrCreate()
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    import sys
    thres = int(sys.argv[2])
    local_thres = None
    main(sys.argv[1], sys.argv[3], sys.argv[4])
    
    