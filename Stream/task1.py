from pyspark import SparkContext
from blackbox import BlackBox
import json
import time
import csv
from collections import Counter
from itertools import combinations
from math import factorial, ceil
import random
import binascii

bx = BlackBox()

def str2int(string):
    return int(binascii.hexlify(string.encode('utf8')),16)

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
    a = random.choice(prime_set)
    b = random.choice(prime_set)
    p = random.choice(prime_set)
    return lambda x: (a*x+b)%m

def myhashs(s):
    result = []
    for layer in bloom_filter:
        result.append(layer.hashfunc(s))
    return result

class BloomFilter_Layer:
    def __init__(self, length):
        self.length = length
        self.hashfunc = gen_hash_func(self.length)
        self.bit_array = [0 for _ in range(self.length)]
    
    def update_bit_array(self, hvalues):
        for hv in hvalues:
            self.bit_array[hv] = 1
        
    def check_familiar(self,u):
        hashidx = self.hashfunc(u)
        return self.bit_array[hashidx]
    
def apply_filter(u):
    uint = str2int(u)
    result = []
    for i,layer in enumerate(bloom_filter):
        result.append((i,layer.hashfunc(uint)))
    return result

def check_FP(u, user_set):
    uint = str2int(u)
    for layer in bloom_filter:
        if not layer.check_familiar(uint):
            return 0
    return int(u not in user_set)

def main(input_path, stream_size, num_of_asks, output_path):
    start_time = time.time()
    stream_size = int(stream_size)
    num_of_asks = int(num_of_asks)
    user_set = set()
    result = {}
    #minint = float('inf')
    #maxint = 0
    for i in range(num_of_asks):
        stream = bx.ask(input_path, stream_size)
        #maxint = max(max([str2int(u) for u in stream]),maxint)
        #minint = min(min([str2int(u) for u in stream]),minint)
        stream_rdd = sc.parallelize(stream)
        FP = stream_rdd.map(lambda x: (1,check_FP(x,user_set))).reduceByKey(lambda a,b: a+b).collect()[0][1]
        result[i] = FP/stream_size
        layer_wise_hashvalues = stream_rdd.flatMap(apply_filter).groupByKey().mapValues(set).collect()
        for j,hvalues in layer_wise_hashvalues:
            bloom_filter[j].update_bit_array(hvalues)
            #print("front:",sum(bloom_filter[j].bit_array[:10]))
            #print("back:",sum(bloom_filter[j].bit_array[-10:]))
        user_set = user_set.union(set(stream))
        print(result[i])
    #print(maxint,minint)
    with open(output_path,'w') as file:
        writer = csv.writer(file)
        writer.writerow(("Time","FPR"))
        for k,v in result.items():
            writer.writerow((k, v))
    print(time.time()-start_time)
    
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
NUM_LAYERS = 3
NUM_BIN = 69997
prime_set = gen_prime_num(NUM_LAYERS*10)
bloom_filter = [BloomFilter_Layer(NUM_BIN) for _ in range(NUM_LAYERS)]
if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])