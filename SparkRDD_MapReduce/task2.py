from pyspark import SparkContext
import json
import time

def task_F(rdd):
    return rdd.reduceByKey(lambda a,b: a+b).sortBy(lambda x: x[1]).map(lambda x: [x[0],x[1]]).take(10)
def customized_partitioner(key):
    return hash(key)

def main(input_path,output_path):
    result = {}
    sc = SparkContext("local","task").getOrCreate()
    rdd = sc.textFile(input_path).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"], 1))
    
    #Default
    start_time = time.time()
    default_rdd = rdd
    task_F(default_rdd)
    exe_time = time.time()-start_time
    num_partitions = default_rdd.getNumPartitions()
    partition_size = default_rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
    result['default'] = {
            'n_partition': num_partitions,
            'n_items': partition_size,
            'exe_time': exe_time
        }
    
    #Customized
    if num_partitions*partition_size[0]<50: # When the size is so small that some hash buckets cannot be filled using customized partitioner
        num_partitions = 1
    else:
        num_partitions = 30
    start_time = time.time()
    new_rdd = rdd.partitionBy(num_partitions, customized_partitioner)
    task_F(new_rdd)
    exe_time = time.time()-start_time
    partition_size = new_rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
    result['customized'] = {
        'n_partition': num_partitions,
        'n_items': partition_size,
        'exe_time': exe_time
    }
    with open(output_path,'w') as file:
        file.write(json.dumps(result))
    print(result)
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1],sys.argv[2])