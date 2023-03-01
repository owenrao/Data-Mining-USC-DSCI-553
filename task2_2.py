from pyspark import SparkContext
import json
import time
import csv
from collections import Counter
from itertools import combinations
from math import factorial, ceil, sqrt
import random
import xgboost as xgb

def rmse(predt, y_test):
    total = 0
    for y_,y in zip(predt,y_test):
        total += (y_ - y)**2
    return float(sqrt(total/len(y_test)))

def csv_to_rdd(train_path):
    return sc.textFile(train_path).zipWithIndex().filter(lambda x: x[1]>0).map(lambda x:tuple(x[0].split(",")))

def json_to_rdd(path):
    return sc.textFile(path).map(lambda line: json.loads(line))
'''
def map_features(pair, user_avg_num, item_avg_num):
    u,i = pair
    u_dict = user_avg_num[u]
    u_avg = u_dict[0]
    u_num = u_dict[1]
    try:
        i_dict = item_avg_num[i]
        i_avg = i_dict[0]
        i_num = i_dict[1]
    except KeyError:
        i_avg = 3
        i_num = 0
    
    return u_avg, i_avg, u_num, i_num
'''
def map_item_features(i, item_info):
    try:
        return item_info[i]
    except KeyError:
        return [0,3.0]

def extract_user_features(u_dict):
    result = []
    keys = ["review_count","useful","funny","cool","fans","average_stars","compliment_hot"]
    for k in keys:
        result.append(float(u_dict[k]))
    result.append(len(u_dict["elite"]))
    return result

def main(folder_path, test_path, output_path):
    start_time = (time.time())
    
    train_rdd = csv_to_rdd(folder_path+"yelp_train.csv").map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x[0])+109)%NUM_PART)
    test_rdd = csv_to_rdd(test_path).partitionBy(NUM_PART,lambda x: (13*hash(x[0])+227)%NUM_PART)#.map(lambda x: ((x[0],x[1]),float(x[2])))
    
    user_info = json_to_rdd(folder_path+"user.json").map(lambda d: (d["user_id"],extract_user_features(d))).collectAsMap()
    item_info = json_to_rdd(folder_path+"business.json").map(lambda d: (d["business_id"],[float(d["review_count"]),float(d["stars"])])).collectAsMap()
    
    train_features = train_rdd.map(lambda x: (item_info[x[0][1]]+user_info[x[0][0]], x[1])).collect()
    test_features = test_rdd.map(lambda x: ((x[0],x[1]), map_item_features(x[1], item_info)+user_info[x[0]])).collect()
    
    X_train = [line[0] for line in train_features] 
    y_train = [line[-1] for line in train_features]
    test_pairs = [line[0] for line in test_features]
    X_test = [line[1] for line in test_features]
    #y_test = [line[-1] for line in test_features]
    
    model = xgb.XGBRegressor(learning_rate = 0.2, random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    #print(rmse(y_pred,y_test))

    with open(output_path,'w') as file:
        writer = csv.writer(file)
        writer.writerow(("user_id", "business_id", "prediction"))
        for pair,pred in zip(test_pairs,y_pred):
            writer.writerow((pair[0],pair[1],pred))
      
    print(time.time()-start_time)
    
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    NUM_PART = 30
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])