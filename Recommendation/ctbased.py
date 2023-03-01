import csv
import pandas as pd
import json
from math import sqrt
import random
import numpy as np

#sc = SparkContext("local[*]","task").getOrCreate()
#sc.setLogLevel("ERROR")
#spark = SparkSession(sc)



class Content_based:
    def __init__(self) -> None:
        self.NUM_PART = 30
        self.FILTER_THRES = 15

    def rmse(self, predt, y_test):
        total = 0
        for y_,y in zip(predt,y_test):
            total += (y_ - y)**2
        return float(sqrt(total/len(y_test)))

    def output_features(train_features, test_features, header):
        header = "user_id,business_id,b_review_count,stars,price_range,acc,rto,rr,rd,wa,os,tv,checkin,photo,u_review_count,fans,average_stars,compliment_hot,tags,elite,tip,rating".split(",")
        with open("mdbased_train.csv",'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for line in train_features:
                writer.writerow(list(line[0])+line[1]+[line[-1]])
                
        with open("mdbased_test.csv",'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for line in test_features:
                writer.writerow(list(line[0])+line[1]+[line[-1]])

    def gen_user_profile(self, user_item_ratings, item_profile):
        def map_user_profile(u_ratings):
            count = 0
            for item,rating in u_ratings.items():
                try:    
                    temp = rating*item_profile[item]
                except KeyError:
                    continue
                if count == 0:
                    user_profile = temp
                else:
                    user_profile += temp
                count += 1
            user_profile /= sum(user_profile)
            return user_profile
        return user_item_ratings.map(lambda x: (x[0],map_user_profile(x[1]))).collectAsMap()

    def gen_item_profiles(self,business_rdd):
        def toBool(x):
            if x is not None:
                return 1
            else:
                return 0
        ambience_list = ['romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual']
        def map_item_profile(i_dict):
            att = i_dict["attributes"]
            if not att:
                att = {}
            att_profile = [toBool(att.get("Alcohol",None)),toBool(att.get("GoodForKids",None))]

            ambience = json.loads(att.get("Ambience","{}").replace("'",'"').replace("F",'f').replace("T",'t'))
            ambience_profile = [int(ambience.get(k, False)) for k in ambience_list]
            result = att_profile + ambience_profile# + category_profile
            
            return np.array(result)
        return business_rdd.map(lambda d: (d["business_id"], map_item_profile(d))).collectAsMap()

    def predict(self, train_rdd, business_rdd, test_rdd):
        user_item_ratings = train_rdd.map(lambda x: (x[0][0],(x[0][1],x[1]))).groupByKey().mapValues(dict)
        item_profile = self.gen_item_profiles(business_rdd)
        user_profile = self.gen_user_profile(user_item_ratings, item_profile)
        self.result = test_rdd.map(lambda x: (x[0],np.dot(user_profile[x[0][0]],item_profile[x[0][1]]),x[1])).collect()
        self.y_pred = [line[-2] for line in self.result]
        self.y_test = [line[-1] for line in self.result]
        self.score = self.rmse(self.y_pred,self.y_test)
        return self.y_pred

    def save_result(self, output_path):
       self.result.to_csv(output_path, columns=["user_id","business_id","prediction"], index=False)
       with open(output_path,'w') as file:
            writer = csv.writer(file)
            writer.writerow(("user_id","business_id","prediction"))
            for line in self.result:
                writer.writerow([*line[0],line[1]])


