import sys
from process_dataset import get_data
from process_dataset import MindDataset
from process_dataset import collect_data
from torch.utils.data import DataLoader
from mind import Mind

data = get_data('../data/samples.txt')
mid, seq, avid, label, lens = collect_data(batch_data=data)
dataset = MindDataset(mid, seq, avid, label, lens)

dataloader = DataLoader(dataset,batch_size=32)
dim = 8
max_len = 20
input_units = 20
output_units = 20
iteration = 1
max_K = 3
p = 2

model = Mind(dim,max_len,input_units,output_units,iteration,max_K,p)

model.train()

for data in dataloader:
    mid, seq, avid, label, lens = data
    output = model(mid,seq,avi==)