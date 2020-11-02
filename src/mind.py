import torch
import torch.nn as nn


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask

def squash(inputs):
    vec_squared_norm = torch.sum(torch.square(inputs),dim=1,keepdim=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * inputs  # element-wise
    return vec_squashed



class Routing(nn.Module):
    def __init__(self,max_len,input_units,output_units,iteration=1,max_K=1):
        super(Routing, self).__init__()
        self.iteration = iteration
        self.max_K = max_K
        self.max_len = max_len
        self.input_units = input_units
        self.output_units = output_units
        self.B_matrix = nn.init.normal_(torch.empty(1,max_K,max_len),mean=0,std=1)
        self.B_matrix.requires_grad=False
        self.S_matrix = nn.init.normal_(torch.empty(self.input_units,self.output_units),mean=0,std=1)



    def forward(self,low_capsule, seq_len):
        ## seq: B * 1
        B,_,embed_size = low_capsule.size()
        assert torch.max(seq_len).item() < self.maxlen
        seq_len_tile = seq_len.repeat(1,self.max_K)
        for i in range(self.iteration):
            mask = sequence_mask(seq_len_tile,self.max_len)
            pad = torch.ones_like(mask,dtype=torch.float32) * (-2 ** 16 + 1)
            B_tile = self.B_matrix.repeat(B,1,1)
            B_mask = torch.where(mask,B_tile,pad)
            W = nn.functional.softmax(B_mask,dim=-1)
            low_capsule_new = torch.einsum('ijk,kl->ijl',(low_capsule,self.S_matrix))
            high_capsule_tmp = torch.matmul(W,low_capsule_new)
            high_capsule = squash(high_capsule_tmp)
            B_delta = torch.sum(
                torch.matmul(high_capsule, torch.transpose(low_capsule_new, dim0=1,dim1=2)),
                dim=0, keepdim=True)
            self.B_matrix += B_delta

        return high_capsule





