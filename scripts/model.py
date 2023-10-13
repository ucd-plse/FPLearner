import torch
from torch import nn
import torch.nn.functional as f
from torch_geometric.nn import global_mean_pool, HeteroConv
from torch_scatter import scatter
import torch_geometric as pyg


class HeteroGNN(nn.Module):
    def __init__(self, graphs, out_channels=100, num_layers=3):
        super().__init__()

        # graphs = ['AST', 'CFG', 'PDG', 'CAST', 'DEP']

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('node', i, 'node'): pyg.nn.GatedGraphConv(out_channels, num_layers) for i in graphs
            }, aggr='sum')

            self.convs.append(conv)

        self.classifier = nn.Linear(in_features=out_channels, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, device):
        edge_index_dict = {}
        for k in data.edge_index_dict:
            edge_index_dict[k] = data.edge_index_dict[k].type(torch.LongTensor)
        data.edge_index_dict = edge_index_dict
        data = data.to(device, non_blocking=True)
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        batch_idx = data.node_stores[0]._mapping['batch']
        rep = global_mean_pool(x_dict['node'], batch_idx) # Average pooling
        rep = self.classifier(rep)
        result = self.sigmoid(rep).squeeze(dim=-1)
        return result


class HeteroRGCN(nn.Module):
    def __init__(self, graphs, out_channels=100, num_layers=3):
        super().__init__()

        # graphs = ['AST', 'CFG', 'PDG', 'CAST', 'DEP']

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('node', i, 'node'): pyg.nn.GCNConv(100, out_channels) for i in graphs
            }, aggr='sum')

            self.convs.append(conv)

        self.classifier = nn.Linear(in_features=out_channels, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, device):
        edge_index_dict = {}
        for k in data.edge_index_dict:
            edge_index_dict[k] = data.edge_index_dict[k].type(torch.LongTensor)
        data.edge_index_dict = edge_index_dict
        data = data.to(device, non_blocking=True)
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        batch_idx = data.node_stores[0]._mapping['batch']
        rep = global_mean_pool(x_dict['node'], batch_idx) # Average pooling
        rep = self.classifier(rep)
        result = self.sigmoid(rep).squeeze(dim=-1)
        return result

