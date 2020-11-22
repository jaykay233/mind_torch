# coding:utf8
fo = open("samples.txt", "w",buffering=True)

with open('transform_data.txt', 'r') as f:
    for line in f.readlines():
        items = line.replace('\n', '').split('\t')
        mid = items[0]
        seq = items[1]
        item_ids = []
        labels = []
        pos_label = items[2]
        item_ids.append(pos_label)
        labels.append(1)
        neg_label = [s for s in items[3].split(',')]
        item_ids.extend(neg_label)
        labels.extend([0]*len(neg_label))
        for i in range(len(labels)):
            line = mid + '\t' + seq + '\t' + item_ids[i] + '\t' + str(labels[i])
            fo.write(line + '\n')

fo.close()


