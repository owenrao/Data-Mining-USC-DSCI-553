from pyspark import SparkContext
import json

def customized_partitioner(key):
    return hash(key)

def main(input_path,output_path):
    sc = SparkContext("local","task").getOrCreate()
    rdd = sc.textFile(input_path).map(lambda x: json.loads(x)).persist()
    user_rdd = rdd.map(lambda x: (x["user_id"], 1)).partitionBy(rdd.getNumPartitions(), customized_partitioner)
    biz_rdd = rdd.map(lambda x: (x["business_id"], 1)).partitionBy(rdd.getNumPartitions(), customized_partitioner)
    
    
    
    user_count = list(reversed(user_rdd.reduceByKey(lambda a,b: a+b).map(lambda x: [x[0],x[1]]).sortBy(lambda x: [-x[1],x[0]],ascending=False).collect()))
    biz_count = list(reversed(biz_rdd.reduceByKey(lambda a,b: a+b).map(lambda x: [x[0],x[1]]).sortBy(lambda x: [-x[1],x[0]],ascending=False).collect()))
    result = {
        "n_review": rdd.count(),
        "n_review_2018": rdd.filter(lambda x: x["date"].startswith("2018")).count(),
        "n_user": user_rdd.distinct().count(),
        "top10_user": user_count[:10],
        "n_business": biz_rdd.distinct().count(),
        "top10_business": biz_count[:10]
    }
    with open(output_path,'w') as file:
        file.write(json.dumps(result))
    print(result)
    return result
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1],sys.argv[2])