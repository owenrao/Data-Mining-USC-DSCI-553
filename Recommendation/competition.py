# Method Description:
# I have built an SVD model at first, which didnt out perform Regression model. As a result, I am sticking with this model and trying to optimize it by adding more features and tweaking with the params.
# Comparing to the original version of the features, I have added a lot: b_review_count,stars,price_range,acc,rto,rr,rd,wa,os,tv,longitude,latitude,is_open,categories_count,checkin,photo,u_review_count,average_stars,tags,compliments,fans,elite,tip
# I have tweaked the parameters using scikit-learn GridSearchCV, which optimizes my current params.

# Error Distribution:
#>=0 and <1: 81607
#>=1 and <2: 45209
#>=2 and <3: 12307
#>=3 and <4: 2902
#>=4 and <5: 19

# RMSE: 0.9783914463141258

# Execution Time: 410.37642335891724

import numpy as np
import pandas as pd
import csv
from pyspark import SparkContext
from pyspark.sql import SparkSession
import json
import time
import random
import xgboost as xgb

from mdbased import Model_based

def rmse(predt, y_test):
    total = 0
    for y_,y in zip(predt,y_test):
        total += (y_ - y)**2
    return float(sqrt(total/len(y_test)))

def toBool(x):
        if x=="True":
            return 1
        else:
            return 0

def csv_to_rdd(train_path):
    return sc.textFile(train_path).zipWithIndex().filter(lambda x: x[1]>0).map(lambda x:tuple(x[0].split(",")))

def json_to_rdd(path):
    return sc.textFile(path).map(lambda line: json.loads(line))

def main(folder_path, test_path, output_path):
    start_time = (time.time())
    
    train_rdd = csv_to_rdd(folder_path+"yelp_train.csv").map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x[0])+109)%NUM_PART)
    test_rdd = csv_to_rdd(test_path).map(lambda x: ((x[0],x[1]),1)).partitionBy(NUM_PART,lambda x: (13*hash(x[0])+227)%NUM_PART)
    checkinz_rdd = json_to_rdd(folder_path+"checkin.json")
    photo_rdd = json_to_rdd(folder_path+"photo.json")
    user_rdd = json_to_rdd(folder_path+"user.json")
    item_rdd = json_to_rdd(folder_path+"business.json")
    tip_rdd = json_to_rdd(folder_path+"tip.json")
    
    model_based = Model_based()
    #model_based.gen_features(train_rdd, test_rdd, user_rdd, item_rdd, checkinz_rdd, photo_rdd, tip_rdd)
    model_based.fit(train_rdd, test_rdd, user_rdd, item_rdd, checkinz_rdd, photo_rdd, tip_rdd)
    #model_based.output_features()
    mb_result = model_based.predict()
    #mb_rmse = model_based.score
    #print(mb_rmse)
    model_based.save_result(output_path)
    print(time.time()-start_time)
    
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
if __name__ == "__main__":
    NUM_PART = 30
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])