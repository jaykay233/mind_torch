import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask


def squash(inputs):
    vec_squared_norm = torch.sum(torch.square(inputs), dim=1, keepdim=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * inputs  # element-wise
    return vec_squashed


class Routing(nn.Module):
    def __init__(self, max_len, input_units, output_units, iteration=1, max_K=1):
        super(Routing, self).__init__()
        self.iteration = iteration
        self.max_K = max_K
        self.max_len = max_len
        self.input_units = input_units
        self.output_units = output_units
        self.B_matrix = nn.init.normal_(torch.empty(1, max_K, max_len), mean=0, std=1)
        self.B_matrix.requires_grad = False
        self.S_matrix = nn.init.normal_(torch.empty(self.input_units, self.output_units), mean=0, std=1)

    def forward(self, low_capsule, seq_len):
        ## seq: B * 1
        ## low_capsule: B * H * D
        global high_capsule
        B, _, embed_size = low_capsule.size()
        assert torch.max(seq_len).item() < self.max_len
        seq_len_tile = seq_len.repeat(1, self.max_K)
        for i in range(self.iteration):
            mask = sequence_mask(seq_len_tile, self.max_len)
            ## mask: B * max_K * max_len
            ## W: B * max_K * max_len
            ## low_capsule_new: B * max_len * hidden_units
            pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
            B_tile = self.B_matrix.repeat(B, 1, 1)
            B_mask = torch.where(mask, B_tile, pad)
            W = nn.functional.softmax(B_mask, dim=-1)
            low_capsule_new = torch.einsum('ijk,kl->ijl', (low_capsule, self.S_matrix))
            high_capsule_tmp = torch.matmul(W, low_capsule_new)
            high_capsule = squash(high_capsule_tmp)
            B_delta = torch.sum(
                torch.matmul(high_capsule, torch.transpose(low_capsule_new, dim0=1, dim1=2)),
                dim=0, keepdim=True)
            self.B_matrix += B_delta
        ## high_capsule: B * max_K * hidden_units
        return high_capsule


class Mind(nn.Module):
    def __init__(self, dim, max_len, input_units, output_units, iteration, max_K, p):
        super(Mind, self).__init__()
        self.dim = dim
        self.max_len = max_len
        self.max_K = max_K
        self.p = p
        self.user_embedding = nn.Embedding(1100, 8, scale_grad_by_freq=True, padding_idx=0)
        self.item_embedding = nn.Embedding(12000, 8, scale_grad_by_freq=True, padding_idx=0)
        self.routing = Routing(max_len, input_units, output_units, iteration, max_K)
        self.label_linear = nn.Linear(dim, output_units)
        self.user_linaer = nn.Linear(dim, output_units)
        self.output_units = output_units
        self.input_units = input_units
        self.final_linear = nn.Linear(self.max_K * self.max_len, 1)

    def forward(self, user_ids, items, labels, seq_lens):
        ## user_ids: B * 1
        ## items: B*H
        ## labels: B*1
        user_ids_embedding = self.user_embedding(user_ids)
        user_ids_embedding = self.user_linaer(user_ids_embedding)
        item_ids_embedding = self.item_embedding(items)
        labels_embedding = self.item_embedding(labels)
        labels_embedding = self.label_linear(labels_embedding)
        ## B * 1 * output_units
        capsule_output = self.routing(item_ids_embedding, seq_lens)
        ## capsule_output_user_added: B * max_K * output_units
        capsule_output_user_added = capsule_output + user_ids_embedding
        capsule_output_user_added = F.relu_(capsule_output_user_added)
        attention_weight = torch.multiply(capsule_output_user_added, labels_embedding.repeat(1, self.max_K, 1))
        attention_weight = torch.pow(attention_weight, self.p)
        attention_weight = torch.sum(attention_weight, dim=-1, keepdim=False)
        attention_weight = nn.functional.softmax(attention_weight, dim=1)
        attention_output = capsule_output_user_added * attention_weight.repeat(1, 1, self.output_units)
        attention_output = attention_output.view(attention_weight.size()[0], -1)
        attention_output = self.final_linear(attention_output)
        output = F.sigmoid(attention_output)
        return output


if __name__ == '__main__':
    batch_size = 3
    seq_length = 3
    dim = 4
    input = torch.rand(batch_size, seq_length, dim)
    seq_len = torch.ones(batch_size, 1)
    routing = Routing(seq_length, 4, 3, 1, 2)
    res = routing(input, seq_len)
    print(res.shape)
