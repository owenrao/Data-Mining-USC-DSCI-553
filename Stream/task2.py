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
    return lambda x: ((a*x+b)%p)%m

def myhashs(s):
    result = []
    for f in hashf_list:
        result.append(f(s))
    return result

def gen_trailing_zero(x):
    #bit_array = []
    xint = str2int(x)
    trailing_zero = 0
    for f in hashf_list:
        hvalue = f(xint)
        #bit_array.append(hvalue)
        if hvalue == 1:
            trailing_zero = 0
        else:
            trailing_zero += 1
    #print(bit_array)
    return trailing_zero
    

def main(input_path, stream_size, num_of_asks, output_path):
    start_time = time.time()
    stream_size = int(stream_size)
    num_of_asks = int(num_of_asks)
    result = []
    sum_gt = 0
    sum_est = 0
    for i in range(num_of_asks):
        stream = bx.ask(input_path, stream_size)
        stream_rdd = sc.parallelize(stream)
        max_trailing_zero = stream_rdd.map(gen_trailing_zero).max()
        estimation = 2**max_trailing_zero
        gt = len(set(stream))
        sum_gt += gt
        sum_est += estimation
        #print(max_trailing_zero)
        result.append((1,stream_size,estimation))
    score = sum_est/sum_gt
    print(score)
    
    with open(output_path,'w') as file:
        writer = csv.writer(file)
        writer.writerow(("Time","Ground Truth","Estimation"))
        for i in result:
            writer.writerow(i)
            
    print(time.time()-start_time)
    
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
NUM_HASH = 100
NUM_BIN = 2
prime_set = gen_prime_num(NUM_HASH*5)
hashf_list = [gen_hash_func(NUM_BIN) for _ in range(NUM_HASH)]
if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])