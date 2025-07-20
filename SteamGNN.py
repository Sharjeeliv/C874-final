import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SteamGNN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, hid_feats))
        for _ in range(num_layers-1):
            self.convs.append(SAGEConv(hid_feats, hid_feats))

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return h

    def decode(self, h, edge_pairs):
        u, v = edge_pairs
        return (h[u] * h[v]).sum(dim=1)