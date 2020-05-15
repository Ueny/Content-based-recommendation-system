from pyspark import SparkContext, SparkConf
import sys
import json
import re
import math
import time

train_file = sys.argv[1]
model_file = sys.argv[2]
stopwords_file = sys.argv[3]

def TF_IDF(freq_list, IDF, word_index):
    max_freq = 0
    for _, freq in freq_list:
        max_freq = max(max_freq, freq)

    word_score = []
    for word, freq in freq_list:
        word_score.append([word_index[word], freq/max_freq * IDF[word]])
    word_list = sorted(word_score, key=lambda s: s[1], reverse=True)[:200]
    return list(dict(word_list).keys())

start_time = time.time()
conf = SparkConf().setAppName('inf553_hw3_task2train').setMaster('local[*]').set('spark.driver.memory', '4G')
sc = SparkContext(conf=conf)

stopwords = sc.textFile(stopwords_file).collect()

review = sc.textFile(train_file) \
    .map(lambda s: json.loads(s))\
    .persist()

N = review.map(lambda s: s['business_id'])\
    .distinct()\
    .count()

business = review.map(lambda s: (s['business_id'], s['text'])) \
    .reduceByKey(lambda s1, s2: s1 + ' ' + s2) \
    .map(lambda s: (s[0], re.sub(r'[^A-Za-z]', ' ', s[1]).lower().split())) \
    .flatMap(lambda s: [(s[0], key) for key in s[1]])\
    .filter(lambda s: s[1] not in stopwords)\
    .persist()

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

word_index = business.map(lambda s: s[0][1])\
    .distinct()\
    .zipWithIndex()\
    .collectAsMap()

IDF = business.map(lambda s: (s[0][1], 1)) \
    .reduceByKey(lambda u, v: u + v) \
    .map(lambda s: (s[0], math.log(N/s[1], 2))) \
    .collectAsMap()

business_profile = business.map(lambda s: (s[0][0], (s[0][1], s[1])))\
    .groupByKey()\
    .mapValues(list)\
    .map(lambda s: (s[0], TF_IDF(s[1], IDF, word_index)))\
    .persist()


# print('t1: ', time.time()-start_time)
#
#
# business_profile = TF.map(lambda s: (s[0], [(key, tf * IDF[key]) for key, tf in s[1]])) \
#     .map(lambda s: (s[0], sorted(s[1], key=lambda x: x[1], reverse=True)[:200])) \
#     .map(lambda s: (s[0], [word_index[key] for key, value in s[1]])) \
#     .persist()
    # .map(lambda s: (s[0], int("".join(['1' if key in s[1] else '0' for key in IDF.keys()]), base=2))) \
    # .persist()\
# print('t2: ', time.time()-start_time)


user_profile = review.map(lambda s: (s['business_id'], s['user_id'])) \
    .distinct() \
    .join(business_profile) \
    .map(lambda s: s[1]) \
    .reduceByKey(lambda s1, s2: list(set(s1) | set(s2)))\
    .collect()
# print('t3: ', time.time()-start_time)
output = dict()
output['user_profile'] = dict(user_profile)
output['business_profile'] = dict(business_profile.collect())
with open(model_file, 'w') as f:
    f.write(json.dumps(output))

end_time = time.time()
print('Duration: ', end_time - start_time)
