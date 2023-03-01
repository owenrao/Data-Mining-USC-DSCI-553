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
from sklearn.metrics import log_loss
from datetime import datetime

def rmse(predt, y_test):
    total = 0
    for y_,y in zip(predt,y_test):
        total += (y_ - y)**2
    return float(sqrt(total/len(y_test)))

def csv_to_rdd(train_path):
    return sc.textFile(train_path).zipWithIndex().filter(lambda x: x[1]>0).map(lambda x:tuple(x[0].split(",")))

def json_to_rdd(path):
    return sc.textFile(path).map(lambda line: json.loads(line))

def extract_user_features(u_dict):
    result = []
    keys = ["review_count","useful","funny","cool","fans","average_stars","compliment_hot"]
    for k in keys:
        result.append(float(u_dict[k]))
    result.append(len(u_dict["elite"]))
    return result

def map_item_features(i, item_info):
    return item_info.get(i,[0,3.0])

def map_corated(ub,corated_itemset,item_user_ratings):
    u,i = ub
    try:
        i_ratings = item_user_ratings[i]
    except KeyError:
        return 0
    result = 0
    for j in corated_itemset:
        j_ratings = item_user_ratings[j]
        corated_set = set(i_ratings.keys()).intersection(set(j_ratings.keys()))
        if len(corated_set)>=CORATED_THRES:
            result += 1
    return result

def gen_pearson_weights(ub,corated_itemset,item_user_ratings):
    u,i = ub
    try:
        i_ratings = item_user_ratings[i]
    except KeyError:
        return {}
    result = {}
    for j in corated_itemset:
        j_ratings = item_user_ratings[j]
        corated_set = set(i_ratings.keys()).intersection(set(j_ratings.keys()))
        if len(corated_set)<CORATED_THRES:
            result[j] = 0
            continue
        i_corated_ratings = [i_ratings[u] for u in corated_set]
        j_corated_ratings = [j_ratings[u] for u in corated_set]
        i_avg,j_avg = calc_avg(corated_set,i_corated_ratings,j_corated_ratings)
        pcc_nominator = 0
        pcc_denominator_i = 0
        pcc_denominator_j = 0
        for rui,ruj in zip(i_corated_ratings,j_corated_ratings):
            rui -= i_avg
            ruj -= j_avg
            pcc_nominator += rui*ruj
            pcc_denominator_i += rui**2
            pcc_denominator_j += ruj**2
        if pcc_nominator==0:
            result[j] = 0
            continue
        pcc_denominator_i = sqrt(pcc_denominator_i)
        pcc_denominator_j = sqrt(pcc_denominator_j)
        wij = pcc_nominator/((pcc_denominator_i*pcc_denominator_j))
        result[j] = wij

    return dict(sorted([(k,v) for k,v in result.items() if v>0],key=lambda x: x[1],reverse=True)[:N_NEIGHBOR])

def main():
    start_time = (time.time())
    
    # Label
    gt_rdd = csv_to_rdd(gt_path).map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x)+109)%NUM_PART)
    label = {}
    with open("model_train/svd_xgb_compare_train.csv") as file:
        reader = csv.reader(file)
        for line in reader:
            label[(line[0],line[1])] = int(line[2])
    with open("model_train/svd_xgb_compare_gt.csv") as file:
        reader = csv.reader(file)
        for line in reader:
            label[(line[0],line[1])] = int(line[2])
    # Feature
    train_rdd = csv_to_rdd(train_path).map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x[0])+109)%NUM_PART)
    user_rated_items = train_rdd.map(lambda x: x[0]).groupByKey().mapValues(set).collectAsMap()
    user_avg_ratings = train_rdd.map(lambda x: (x[0][0],(x[1],1))).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])).mapValues(lambda v: v[0]/v[1]).collectAsMap()
    item_user_ratings = train_rdd.map(lambda x: (x[0][1],(x[0][0],x[1]))).groupByKey().mapValues(dict).collectAsMap()
    count_neighbors = gt_rdd.map(lambda ub: (ub[0],len(gen_pearson_weights(ub, user_rated_items[ub[0][0]], item_user_ratings)))).collectAsMap()
    user_info = json_to_rdd(folder_path+"user.json").map(lambda line: (line["user_id"], [float(line["review_count"]), float(line["average_stars"]), float(line["useful"]), float(line["fans"]), float(line["cool"]),
                                                 float(line["funny"]), float(line["compliment_writer"]), float(line["compliment_funny"]), float(line["compliment_cool"]),
                                                 float(line["compliment_hot"]), float(line["compliment_more"]), float(line["compliment_profile"]), float(line["compliment_cute"]),
                                                 float(line["compliment_list"]), float(line["compliment_note"]), float(line["compliment_plain"]), float(line["compliment_photos"]),
                                                 len(line["elite"].split(", ")), 2022 - datetime.strptime(line["yelping_since"], "%Y-%m-%d").year])).collectAsMap()
    item_info = json_to_rdd(folder_path+"business.json").map(lambda d: (d["business_id"],[float(d["review_count"]),float(d["stars"])])).collectAsMap()
    train_features = train_rdd.map(lambda x: (x[0], map_item_features(x[0][1], item_info)+user_info[x[0][0]], label[x[0]])).collect() #[count_neighbors.get(x[0],0)]
    test_features = gt_rdd.map(lambda x: (x[0], map_item_features(x[0][1], item_info)+user_info[x[0][0]], label[x[0]])).collect()
    # Must use np array here, needs array.shape in model.fit
    X_train = np.array([line[1] for line in train_features])
    y_train = np.array([line[-1] for line in train_features])

    X_test = np.array([line[1] for line in test_features])
    y_test = np.array([line[-1] for line in test_features])

    model = xgb.XGBRegressor(random_state=0)
    model.fit(X_train,y_train)
    with open("model_train/model.md",'wb') as file:
        pickle.dump(model,file)
    print(time.time()-start_time)
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_pred = np.array([int(i>=0.5) for i in y_pred])

    with open("model_train/y_pred.csv",'w') as file:
        for pred in y_pred:
            file.write(str(pred))
            file.write("\n")
    with open("model_train/y_truth.csv",'w') as file:
        for pred in y_test:
            file.write(str(pred))
            file.write("\n")
    print(sum(y_pred==y_test)/len(y_test))
            
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    NUM_PART = 30
    CORATED_THRES = 25
    N_NEIGHBOR = 20
    train_test_border = 0.8
    folder_path = "../resource/asnlib/publicdata/"
    train_path = folder_path+"yelp_train.csv"
    gt_path = "SVD_train/popular_val_15.csv"
    import sys
    main()