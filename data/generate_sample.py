datas = []

out = open('samples.txt','w')
with open('transformed_data.txt','r') as f:
    for line in f.readlines():
        items = line.replace('\n','').split('\t')
        mid = items[0]
        seq = items[1]
        pos_label = items[2]
        line = mid + '\t' + seq + '\t' + pos_label + '\t' + '1'
        out.write(line + '\n')
        neg_label = [s for s in items[3].split(',')]
        for neg in neg_label:
            line = mid + '\t' + seq + '\t' + neg + '\t' + '0'
            out.write(line + '\n')

out.close()
        

