from pyspark import SparkContext, SparkConf
import sys
import json
import re
import math
import time

train_file = 'train_review.json'
model_file = 'task2.model'
stopwords_file = 'stopwords'

start_time = time.time()
conf = SparkConf().setAppName('inf553_hw3_task2train').setMaster('local[*]').set('spark.driver.memory', '4G')
sc = SparkContext(conf=conf)

stopwords = sc.textFile(stopwords_file).collect()

review = sc.textFile(train_file) \
    .map(lambda s: json.loads(s))\
    .persist()

business = review.map(lambda s: (s['business_id'], s['text'])) \
    .reduceByKey(lambda s1, s2: s1 + ' ' + s2) \
    .map(lambda s: (s[0], re.sub(r'[^A-Za-z]', ' ', s[1]).lower().split())) \
    .flatMap(lambda s: [(s[0], key) for key in s[1]])\
    .filter(lambda s: s[1] not in stopwords)\
    .persist()
    # .map(lambda s: (s[0], [key for key in s[1] if key not in stopwords])) \


# figure out rare words
total_num = business.count()

rare_words = business.map(lambda s: (s[1], 1))\
    .reduceByKey(lambda u, v: u + v) \
    .filter(lambda s: s[1] < total_num * 0.000001)\
    .persist()

business = business.map(lambda s: (s[1], s[0]))\
    .subtractByKey(rare_words)\
    .map(lambda s: ((s[1], s[0]), 1))\
    .reduceByKey(lambda u, v: u + v)\
    .persist()

TF = business.map(lambda s: (s[0][0], (s[0][1], s[1])))\
    .groupByKey()\
    .mapValues(list)\
    .map(lambda s: (s[0], dict(s[1])))\
    .map(lambda s: (s[0], s[1], max(s[1].values())))\
    .map(lambda s: (s[0], [(key, cnt / s[2]) for key, cnt in s[1].items()])) \
    .persist()

N = TF.count()

IDF = business.map(lambda s: (s[0][1], 1)) \
    .reduceByKey(lambda u, v: u + v) \
    .map(lambda s: (s[0], math.log(N/s[1], 2))) \
    .sortByKey()\
    .collect()
IDF = dict(IDF)

business_profile = TF.map(lambda s: (s[0], [(key, tf * IDF[key]) for key, tf in s[1]])) \
    .map(lambda s: (s[0], sorted(s[1], key=lambda x: x[1], reverse=True)[:200])) \
    .map(lambda s: (s[0], [key for key, value in s[1]])) \
    .map(lambda s: (s[0], int("".join(['1' if key in s[1] else '0' for key in IDF.keys()]), base=2))) \
    .persist()\

user_profile = review.map(lambda s: (s['business_id'], s['user_id'])) \
    .distinct() \
    .join(business_profile) \
    .map(lambda s: s[1]) \
    .reduceByKey(lambda s1, s2: s1 | s2)\
    .collect()

output = dict()
output['business_profile'] = dict(business_profile.collect())
output['user_profile'] = dict(user_profile)
with open(model_file, 'w') as f:
    f.write(json.dumps(output))

end_time = time.time()
print('Duration: ', end_time - start_time)
