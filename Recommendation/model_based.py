import json
import time
import csv
import pandas as pd
from math import sqrt
import random
import xgboost as xgb
import numpy as np

#sc = SparkContext("local[*]","task").getOrCreate()
#sc.setLogLevel("ERROR")
#spark = SparkSession(sc)

class Model_based:
    def __init__(self) -> None:
        self.NUM_PART = 30
        self.FILTER_THRES = 15

    def rmse(self, predt, y_test):
        total = 0
        for y_,y in zip(predt,y_test):
            total += (y_ - y)**2
        return float(sqrt(total/len(y_test)))

    def output_features(self):
        header = "user_id,business_id,b_review_count,stars,price_range,acc,rto,rr,rd,wa,os,tv,is_open,longtitude,latittude,categories_count,checkin,photo,u_review_count,average_stars,tags,compliments,fans,elite,tip,rating".split(",")
        self.train_features.to_csv("mdbased_train.csv", columns=header, index=False)
                
        self.test_features.to_csv("mdbased_test.csv", columns=header, index=False)

    def gen_features(self, train_rdd, test_rdd, user_rdd, business_rdd, checkin_rdd, photo_rdd, tip_rdd):
        def map_item_features(i):
            return item_info.get(i,[np.nan for _ in range(14)])+[biz_checkin.get(i,np.nan), biz_photo.get(i,np.nan)]

        def map_user_features(u):
            return user_info[u]

        def extract_user_features(u_dict):
            result = [float(u_dict['review_count']), float(u_dict['average_stars']),\
                      int(u_dict['useful'])+int(u_dict['funny'])+int(u_dict['cool']), \
                      sum([int(u_dict[k]) for k in u_dict.keys() if k.startswith("compliment")]),\
                      int(u_dict['fans']), int(len(u_dict["elite"]) if u_dict["elite"] is not None else "0")]  
            return result

        def extract_item_features(i_dict):
            def toBool(x):
                if x=="True":
                    return 1
                else:
                    return 0
            result = [float(i_dict["review_count"]),float(i_dict["stars"])]
            att = i_dict["attributes"]
            if not att:
                att = {}
            result += [int(att.get("RestaurantsPriceRange2",2)), toBool(att.get("BusinessAcceptsCreditCards",np.nan)),\
                        toBool(att.get("RestaurantsTakeOut",np.nan)), toBool(att.get("RestaurantsReservations",np.nan)),\
                        toBool(att.get("RestaurantsDelivery",np.nan)), toBool(att.get("WheelchairAccessible",np.nan)), \
                        toBool(att.get("OutdoorSeating",np.nan)), toBool(att.get("HasTV",np.nan)), int(i_dict['is_open']), \
                        (float(i_dict["longitude"])+180)/360 if i_dict["longitude"] is not None else 0.5, \
                        (float(i_dict["latitude"])+90)/180 if i_dict["latitude"] is not None else 0.5, 
                        int(len(i_dict["categories"]) if i_dict["categories"] is not None else 0)]
            return result
        biz_checkin = checkin_rdd.map(lambda d: (d["business_id"],len(list(d["time"].values())))).collectAsMap()
        biz_photo = photo_rdd.map(lambda d: (d["business_id"],d["photo_id"])).groupByKey().mapValues(lambda x: len(x)).collectAsMap()
        biz_tip = tip_rdd.map(lambda d:((d['user_id'],d['business_id']), d["likes"])).reduceByKey(lambda a,b: a+b).collectAsMap()
        #self.user_rate_list = self.train_rdd.map(lambda x: (x[0][0],x[1])).groupByKey().mapValues(list).collectAsMap()
        #self.item_rate_list = self.train_rdd.map(lambda x: (x[0][1],x[1])).groupByKey().mapValues(list).collectAsMap()
        
        user_info = user_rdd.map(lambda d:(d['user_id'],extract_user_features(d))).collectAsMap()
        item_info = business_rdd.map(lambda d:(d['business_id'],extract_item_features(d))).collectAsMap()
        header = "user_id,business_id,b_review_count,stars,price_range,acc,rto,rr,rd,wa,os,tv,is_open,longtitude,latittude,categories_count,checkin,photo,u_review_count,average_stars,tags,compliments,fans,elite,tip,rating".split(",")
        self.train_features = train_rdd.map(lambda x: (x[0][0],x[0][1], *map_item_features(x[0][1]), *map_user_features(x[0][0]), biz_tip.get(x[0], 0), x[1])).toDF(header).toPandas()
        self.test_features = test_rdd.map(lambda x: (x[0][0],x[0][1], *map_item_features(x[0][1]), *map_user_features(x[0][0]), biz_tip.get(x[0], 0), x[1])).toDF(header).toPandas()
    
    def gen_datasets(self):
        self.X_train = self.train_features.iloc[:,2:-1]#[line[1] for line in train_features] 
        self.y_train = self.train_features.iloc[:,-1]#[line[-1] for line in train_features]
        self.test_pairs = self.test_features.iloc[:,:2]#[line[0] for line in test_features]
        self.X_test = self.test_features.iloc[:,2:-1]#[line[1] for line in test_features]
        #self.y_test = self.test_features.iloc[:,-1]

    def fit(self, train_rdd, test_rdd, user_rdd, business_rdd, checkin_rdd, photo_rdd, tip_rdd, n_estimators=500, random_state=0, learning_rate=0.13, max_depth=5):
        self.train_rdd = train_rdd
        self.test_rdd = test_rdd
        self.gen_features(train_rdd, test_rdd, user_rdd, business_rdd, checkin_rdd, photo_rdd, tip_rdd)
        self.gen_datasets()
        self.model = xgb.XGBRegressor(objective = 'reg:linear', learning_rate=0.1, max_depth=5, n_estimators=700, reg_lambda=1.5, n_jobs=-1)#(n_estimators=n_estimators, random_state=random_state, learning_rate=learning_rate, max_depth=max_depth)
        self.model.fit(self.X_train,self.y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        #self.score = self.rmse(self.y_pred,self.y_test)
        self.result = self.test_pairs
        self.result["prediction"] = self.y_pred
        #self.result["label"] = self.y_test
        return self.result

    def save_result(self, output_path):
       
       self.result[["user_id","business_id","prediction"]].to_csv(output_path, index=False)


