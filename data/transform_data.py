import sys
from random import sample
file = sys.argv[1]
id_user_mapping = sys.argv[2]
id_avid_mapping = sys.argv[3]

id2mid = {}
id2avid = {}
mid2id = {}
avid2id = {}
total_list = list(range(11500))
with open(id_user_mapping, 'r') as f:
    for line in f.readlines():
        id, mid = line.replace('\n', '').split('\t')
        id = int(id)
        mid = int(mid)
        if id not in id2mid:
            id2mid[id] = mid
        if mid not in mid2id:
            mid2id[mid] = id

with open(id_avid_mapping, 'r') as f:
    for line in f.readlines():
        id, avid = line.replace('\n', '').split('\t')
        id = int(id)
        avid = int(avid)
        if id not in id2avid:
            id2avid[id] = avid
        if avid not in avid2id:
            avid2id[avid] = id

transformed_file = open('transform_data.txt','w')

with open(file, 'r') as f:
    for line in f.readlines():
        mid, avid_list = line.replace('\n', '').split('\t')
        id = mid2id[int(mid)]
        avid_lists = [int(c) for c in avid_list.split(',')]
        avid_lists = list(map(lambda x: avid2id[x], avid_lists))
        neg_avids = sample(total_list,20)
        neg_list = []
        for c in neg_avids:
            if c not in avid_lists:
                neg_list.append(c)
        if len(neg_list)<10:
            while len(neg_list) < 10:
                neg_list = neg_list + neg_list
        neg_list = neg_list[:10]
        s = str(id) + '\t' + ','.join(str(avid) for avid in avid_lists[:-1]) + '\t' + str(avid_lists[-1]) + '\t' + ','.join(str(c) for c in neg_list)
        transformed_file.write(s+'\n')
