from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set
from graphframes import GraphFrame
import json
import time
import csv
from collections import Counter
from itertools import combinations
from math import factorial, ceil
import random

def csv_to_rdd(path):
    return sc.textFile(path)\
            .zipWithIndex().filter(lambda x: x[1]>0)\
            .map(lambda x:tuple(x[0].split(",")))

def gen_edges(rdd):
    corated_userset = rdd.groupByKey().mapValues(set)
    edges_rdd = corated_userset.flatMap(lambda x: [(tuple(sorted(list(pair))),1) for pair in combinations(x[1],2)])\
                        .reduceByKey(lambda a,b: a+b)\
                        .filter(lambda x: x[1]>=THRES)\
                        .map(lambda x: x[0]).flatMap(lambda x: [x,(x[1],x[0])])
    return edges_rdd

def main(input_path, output_path):
    
    raw_rdd = csv_to_rdd(input_path).map(lambda x: (x[1],x[0])).partitionBy(NUM_PART,lambda x: (23*hash(x)+109)%NUM_PART)
    edges_rdd = gen_edges(raw_rdd)
    edges_df = edges_rdd.toDF(["src","dst"])
    vertices_df = edges_rdd.map(lambda x: (x[0],)).distinct().toDF(["id"])
    G = GraphFrame(vertices_df, edges_df)
    LPA_df = G.labelPropagation(maxIter=5)
    start_time = (time.time())
    result = LPA_df.rdd.map(lambda x: (x[1],x[0])).groupByKey().mapValues(lambda x: sorted(x,reverse=False)).sortBy(lambda x: [len(x[1]),x[1][0]]).collect()
    with open(output_path,'w') as file:
        for label,comm in result:
            file.write(str(comm)[1:-1]+"\n")
    print(time.time()-start_time)

sc = SparkContext("local[*]","task").getOrCreate()
spark = SparkSession(sc)
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    NUM_PART = 30
    
    import sys
    THRES = int(sys.argv[1])
    main(sys.argv[2], sys.argv[3])