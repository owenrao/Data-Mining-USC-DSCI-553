from pyspark import SparkContext
import json
import time
import csv
from collections import Counter
from itertools import combinations
from math import factorial, ceil, sqrt
import random

def csv_to_rdd(train_path):
    return sc.textFile(train_path).zipWithIndex().filter(lambda x: x[1]>0).map(lambda x:tuple(x[0].split(",")))

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

def case_amplification(w):
    w_ = w*abs(w)**1.2
    return w_

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

    return dict(sorted([(k,v) for k,v in result.items() if v>0],key=lambda x: x[1],reverse=True)[:N_NEIGHBOR]) # 

def predict(pair,pcc_weights,item_user_ratings,user_avg_ratings):
    u,i = pair
    nominator = 0
    denominator = 0
    for j,wij in pcc_weights.items():
        ruj = item_user_ratings[j][u]
        nominator += ruj*wij
        denominator += abs(wij)
    if denominator == 0:
        result = user_avg_ratings[u]
    else:
        result = nominator/denominator
    return result

def calc_rmse(result,gt):
    N = len(result)
    rmse = 0
    for line in result:
        pair,r = line
        rmse += (r-gt[pair])**2
    rmse = sqrt(rmse/N)
    return rmse

def main(train_path, test_path, output_path):
    start_time = (time.time())

    train_rdd = csv_to_rdd(train_path).map(lambda x: ((x[0],x[1]),float(x[2]))).partitionBy(NUM_PART,lambda x: (23*hash(x[0])+109)%NUM_PART)
    user_rated_items = train_rdd.map(lambda x: x[0]).groupByKey().mapValues(set).collectAsMap()
    user_avg_ratings = train_rdd.map(lambda x: (x[0][0],(x[1],1))).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])).mapValues(lambda v: v[0]/v[1]).collectAsMap()
    item_user_ratings = train_rdd.map(lambda x: (x[0][1],(x[0][0],x[1]))).groupByKey().mapValues(dict).collectAsMap()
    
    test_rdd = csv_to_rdd(test_path).map(lambda x: (x[0],x[1])).partitionBy(NUM_PART,lambda x: (13*hash(x)+227)%NUM_PART)
    pearson_weights = test_rdd.map(lambda ub: (ub,gen_pearson_weights(ub, user_rated_items[ub[0]], item_user_ratings)))
    pred_result = pearson_weights.map(lambda ub_w: (ub_w[0],predict(ub_w[0],ub_w[1],item_user_ratings,user_avg_ratings))).collect()
    
    gt = csv_to_rdd(test_path).map(lambda x: ((x[0],x[1]),float(x[2]))).collectAsMap()
    print(calc_rmse(pred_result,gt))
    
    ##Output
    with open(output_path,'w') as file:
        writer = csv.writer(file)
        writer.writerow(("user_id", "business_id", "prediction"))
        for pair, pred in pred_result:
            writer.writerow((pair[0], pair[1], pred))
    #print(time.time()-start_time)
    
sc = SparkContext("local[*]","task").getOrCreate()
sc.setLogLevel("ERROR")
if __name__ == "__main__":
    NUM_PART = 30
    CORATED_THRES = 25
    N_NEIGHBOR = 25
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])