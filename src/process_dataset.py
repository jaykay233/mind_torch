import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch


def get_data(filename, encoding='utf8'):
    datas = []
    with open(filename, 'r', encoding=encoding) as f:
        for line in f.readlines():
            mid, seq, avid, label = line.replace('\n', '').split('\t')
            datas.append([mid, seq, avid, label])
    return datas


def transform(string_list, split_char=','):
    return [int(c) for c in string_list.split(split_char) if c != '']


def truncate(int_list, max_len=20, pad=0):
    if len(int_list) < max_len:
        return len(int_list), [pad] * (max_len - len(int_list)) + int_list
    return max_len, int_list[-max_len:]


def collect_data(batch_data, pad=0, max_len=20):
    mid = [c[0] for c in batch_data]
    seq = [c[1] for c in batch_data]
    avid = [c[2] for c in batch_data]
    label = [c[3] for c in batch_data]
    mids = [int(c) for c in mid]
    avids = [int(c) for c in avid]
    labels = [int(c) for c in label]
    seqs = [truncate(transform(c.replace('\t', ''), split_char=','))[1] for c in seq]
    lens = [truncate(transform(c.replace('\t', ''), split_char=','))[0] for c in seq]
    print(len(lens))
    mids = torch.LongTensor(mids)
    seqs = torch.LongTensor(seqs)
    avids = torch.LongTensor(avids)
    labels = torch.LongTensor(labels)
    return mids, seqs, avids, labels, lens


class MindDataset(Dataset):
    def __init__(self, mid, seq, avid, label, lens):
        self.mid = mid
        self.seq = seq
        self.avid = avid
        self.label = label
        self.lens = lens

    def __getitem__(self, item):
        return self.mid[item], self.seq[item], self.avid[item], self.label[item], self.lens[item]

    def __len__(self):
        return len(self.mid)


if __name__ == '__main__':
    data = get_data('../data/samples.txt')
    # print(data)
    mid, seq, avid, label, lens = collect_data(batch_data=data)
    dataset = MindDataset(mid, seq, avid, label, lens)
    print(dataset[0])