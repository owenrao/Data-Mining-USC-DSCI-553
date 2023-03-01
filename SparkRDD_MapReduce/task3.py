from pyspark import SparkContext
import json
import time

N = 6

def customized_partitioner(key):
    return hash(key)

def average(rdd,key,value):
    rdd = rdd.map(lambda x: (x[key], (x[value],1))).partitionBy(N, customized_partitioner)
    return rdd.reduceByKey(lambda x,y: ((x[1]*x[0]+y[1]*y[0])/(x[1]+y[1]),x[1]+y[1])).mapValues(lambda x: x[0])

def lookup(l,key):
    for i in l:
        if i[0] == key:
            return i

def main(rev_input_path,biz_input_path,output_path_a,output_path_b):
    '''def city_stars_func(x):
        return x["city"], lookup(biz_list,x["business_id"])[1]'''
    #A
    sc = SparkContext("local","task").getOrCreate()
    rev_rdd = sc.textFile(rev_input_path).map(lambda x: json.loads(x))
    biz_stars = average(rev_rdd, "business_id", "stars")
    biz_rdd = sc.textFile(biz_input_path).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"],x["city"])).partitionBy(N, customized_partitioner)
    joined = biz_stars.join(biz_rdd).map(lambda x: (x[1][1],x[1][0]))
    city_stars = average(joined,0,1)
    a_result = reversed(city_stars.sortBy(lambda x: [-x[1],x[0]],ascending=False).collect())
    with open(output_path_a,'w') as file:
        file.write("city,stars\n")
        for item in a_result:
            file.write(f"{item[0]},{item[1]}\n")
    
    #B
    time_measure = {}
    
    ## m1
    t1 = time.time()
    m1_result = city_stars.collect()
    m1_result.sort(reverse=False,key=lambda x: [-x[1],x[0]])
    print(m1_result)
    time_measure["m1"] = time.time()-t1
    
    ## m2
    t1 = time.time()
    print(reversed(city_stars.sortBy(lambda x: [-x[1],x[0]],ascending=False).take(10)))
    time_measure["m2"] = time.time()-t1
    
    time_measure["reason"] = "Because in spark, partitions are sorted within itself, then combined and sorted together, which is slower?"
    with open(output_path_b,'w') as file:
        file.write(json.dumps(time_measure))
        
if __name__ == "__main__":
    import sys
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])