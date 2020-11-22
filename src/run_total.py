import sys
from process_dataset import get_data
from process_dataset import MindDataset
from process_dataset import collect_data
from torch.utils.data import DataLoader
from mind import Mind
import torch.nn.functional as F
import torch

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

optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
model.train()

train_iteration = 3

for i in range(train_iteration):
    losses = []
    for data in dataloader:
        mid, seq, label_avid, label, lens = data
        output = model(mid,seq,label_avid,lens)
        output = output.squeeze(dim=-1).float()
        label = label.float()
        loss = F.binary_cross_entropy_with_logits(output,label)
        losses.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    print("epoch {} loss: ".format(i),sum(losses)/len(losses))

