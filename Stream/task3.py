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

def main(input_path, stream_size, num_of_asks, output_path):
    random.seed(553)
    start_time = time.time()
    stream_size = int(stream_size)
    num_of_asks = int(num_of_asks)
    reservoir = []
    result = []
    count = 0
    for i in range(num_of_asks):
        stream = bx.ask(input_path, stream_size)
        if i == 0:
            reservoir = stream
            count = 100
        else:
            for item in stream:
                count += 1
                p = float(SAMPLE_SIZE/count)
                if random.random() < p:#random.choices([0,1], weights=(1-p,p),k=1)[0]: # Sampled
                    rand_idx = random.randint(0,99)
                    reservoir[rand_idx] = item
        current_sample = reservoir
        result.append((count,current_sample[0],current_sample[20],current_sample[40],current_sample[60],current_sample[80]))  
        
    with open(output_path,'w') as file:
        writer = csv.writer(file)
        writer.writerow(("seqnum","0_id","20_id","40_id","60_id","80_id"))
        for i in result:
            writer.writerow(i)
            
    print(time.time()-start_time)
    
SAMPLE_SIZE = 100

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])