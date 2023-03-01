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


def calc_avg(corated_set,i_ratings,j_ratings):
    i_sum = 0
    j_sum = 0
    size = len(corated_set)
    for rui,ruj in zip(i_ratings,j_ratings):
        i_sum += rui
        j_sum += ruj
    i_avg = i_sum/size
    j_avg = j_sum/size
    return i_avg,j_avg

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

    return dict(sorted([(k,v) for k,v in result.items() if v>P_THRES],key=lambda x: x[1],reverse=True)[:N_NEIGHBOR]) # 

def predict(pair,pcc_weights,item_user_ratings,user_avg_ratings,mb_result_map):
    u,i = pair
    nominator = 0
    denominator = 0
    for j,wij in pcc_weights.items():
        ruj = item_user_ratings[j][u]
        nominator += ruj*wij
        denominator += abs(wij)
    if denominator == 0:
        result = mb_result_map[(u,i)]
    else:
        result = nominator/denominator
    return result
    
    
def main(folder_path, test_path, output_path):
    ustart_time = (time.time())
    
    # Feature Extract
    start_time = (time.time())
    train_rdd = csv_to_rdd(folder_path+"yelp_train.csv").map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x[0])+109)%NUM_PART)
    user_info = json_to_rdd(folder_path+"user.json").map(lambda d: (d["user_id"],extract_user_features(d))).collectAsMap()
    item_info = json_to_rdd(folder_path+"business.json").map(lambda d: (d["business_id"],[float(d["review_count"]),float(d["stars"])])).collectAsMap()
    print(time.time()-start_time)
    
    # Model Based Train
    start_time = (time.time())
    train_features = train_rdd.map(lambda x: (item_info[x[0][1]]+user_info[x[0][0]], x[1])).collect()
    X_train = [line[0] for line in train_features] 
    y_train = [line[-1] for line in train_features]
    mb_model = xgb.XGBRegressor(learning_rate = 0.2, random_state=0)
    mb_model.fit(X_train,y_train)
    print(time.time()-start_time)
    
    # Test Feature Extract
    start_time = (time.time())
    test_rdd = csv_to_rdd(test_path).partitionBy(NUM_PART,lambda x: (13*hash(x[0])+227)%NUM_PART)
    test_features = test_rdd.map(lambda x: ((x[0],x[1]), map_item_features(x[1], item_info)+user_info[x[0]])).collect()
    test_pairs = [line[0] for line in test_features]
    X_test = [line[1] for line in test_features]
    mb_result = mb_model.predict(X_test)
    mb_result_map = {k: v for k,v in zip(test_pairs,mb_result)}
    #y_test = [line[-1] for line in test_features]
    print(time.time()-start_time)
    
    # Item Based Pred
    start_time = (time.time())
    #ib_result = Item_based_pred(train_rdd,test_rdd.filter(lambda x: selection_map[x]==0))
    user_rated_items = train_rdd.map(lambda x: x[0]).groupByKey().mapValues(set).collectAsMap()
    user_avg_ratings = train_rdd.map(lambda x: (x[0][0],(x[1],1))).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])).mapValues(lambda v: v[0]/v[1]).collectAsMap()
    item_user_ratings = train_rdd.map(lambda x: (x[0][1],(x[0][0],x[1]))).groupByKey().mapValues(dict).collectAsMap()
    pearson_weights = test_rdd.map(lambda ub: (ub,gen_pearson_weights(ub, user_rated_items[ub[0]], item_user_ratings)))
    ult_result = pearson_weights.map(lambda ub_w: (ub_w[0],predict(ub_w[0],ub_w[1],item_user_ratings,user_avg_ratings,mb_result_map))).collect()
    print(time.time()-start_time)
       
    #ult_result = ib_result+mb_result
    with open(output_path,'w') as file:
        writer = csv.writer(file)
        writer.writerow(("user_id", "business_id", "prediction"))
        for pair,pred in ult_result:
            writer.writerow((pair[0],pair[1],pred))
      
    print(time.time()-ustart_time)
    
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    NUM_PART = 30
    CORATED_THRES = 35
    P_THRES = 0.6
    N_NEIGHBOR = 20
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])