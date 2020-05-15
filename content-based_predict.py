from pyspark import SparkContext, SparkConf
import sys
import json
import math
import time

test_file = 'test_review.json'#sys.argv[1]
model_file = 'task2.model'#sys.argv[2]
output_file = 'task2.predict'#sys.argv[3]

def cosine_dist(iterator, user_profile, business_profile):
    for u_id, b_id in iterator:
        if u_id not in user_profile.keys() or b_id not in business_profile.keys():
            continue
        u_pf = user_profile[u_id]
        b_pf = business_profile[b_id]
        numerator = len(set(u_pf) & set(b_pf))# bin(u_pf & b_pf).count('1')
        denominator = math.log(len(u_pf)) * math.log(len(b_pf))
        similarity = numerator/denominator
        if similarity >= 0.01:
            yield [['user_id', u_id], ['business_id', b_id], ['sim', similarity]]

start_time = time.time()
conf = SparkConf().setAppName('inf553_hw3_task2predict').setMaster('local[*]')
sc = SparkContext(conf=conf)

with open(model_file, 'r') as f:
    profiles = json.load(f)

user_profile = profiles['user_profile']
business_profile = profiles['business_profile']

test_review = sc.textFile(test_file)\
    .map(lambda s: json.loads(s))\
    .map(lambda s: (s['user_id'], s['business_id']))\
    .map(lambda s: (s[0], s[1]))\
    .mapPartitions(lambda s: cosine_dist(iterator=s, user_profile=user_profile, business_profile=business_profile))\
    .map(lambda s: dict(s))\
    .collect()

with open(output_file, 'w') as f:
    for line in test_review:
        f.write(json.dumps(line) + '\n')

print('Duration: ', time.time()-start_time)