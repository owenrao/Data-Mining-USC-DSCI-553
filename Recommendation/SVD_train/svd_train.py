from pyspark import SparkContext
import json
import time
import csv
from collections import Counter
from itertools import combinations
from math import factorial, ceil, sqrt
import random
import xgboost as xgb
import pickle
# must import np for sklearn classifier fit to work
import numpy as np
import pandas as pd

def rmse(predt, y_test):
    total = 0
    for y_,y in zip(predt,y_test):
        total += (y_ - y)**2
    return float(sqrt(total/len(y_test)))

def csv_to_rdd(train_path):
    return sc.textFile(train_path).zipWithIndex().filter(lambda x: x[1]>0).map(lambda x:tuple(x[0].split(",")))

def json_to_rdd(path):
    return sc.textFile(path).map(lambda line: json.loads(line))

def filter_frequent():
    return

def main():
    start_time = (time.time())
    
    # Feature
    train_rdd = csv_to_rdd(train_path).map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x[0])+109)%NUM_PART)
    #user_rated_items = train_rdd.map(lambda x: x[0]).groupByKey().mapValues(set).collectAsMap()
    item_rated_count = train_rdd.map(lambda x: (x[0][1],x[0][0])).groupByKey().mapValues(len).collectAsMap()
    SVD_train_data = train_rdd.filter(lambda x: item_rated_count.get(x[0][1],0) >= FILTER_THRES).collect()
    with open(f"SVD_train/popular_{FILTER_THRES}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(('user_id','business_id','ratings'))
        for ub,r in SVD_train_data:
            writer.writerow((ub[0],ub[1],r))
    
    test_rdd = csv_to_rdd(test_path).map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x[0])+109)%NUM_PART)
    SVD_test_data = test_rdd.filter(lambda x: item_rated_count.get(x[0][1],0) >= FILTER_THRES).collect()
    with open(f"SVD_train/popular_val_{FILTER_THRES}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(('user_id','business_id','ratings'))
        for ub,r in SVD_test_data:
            writer.writerow((ub[0],ub[1],r))
    
    
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    NUM_PART = 30
    CORATED_THRES = 25
    N_NEIGHBOR = 20
    FILTER_THRES = 20
    train_test_border = 0.8
    folder_path = "../resource/asnlib/publicdata/"
    train_path = folder_path+"yelp_train.csv"
    test_path = folder_path+"yelp_val.csv"
    import sys
    main()