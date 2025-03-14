import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class GQN(nn.Module):
    def __init__(self, input_dim, global_input_dim, hidden_dim, num_layers=2):
        super(GQN, self).__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList([pyg_nn.TransformerConv(input_dim, hidden_dim)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.TransformerConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.fc = nn.Linear(hidden_dim+global_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)
    
    def forward(self, batch, global_features):
        node_features, edge_index = batch.x, batch.edge_index
        x = node_features
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Repeat each tensor value based on the number of nodes in each graph
        num_nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]  # Number of nodes per graph
        repeated_tensor = torch.cat([global_features[i].repeat(n, 1) for i, n in enumerate(num_nodes_per_graph)], dim=0)

        # Concatenate the repeated tensor with node features
        x = torch.cat([x, repeated_tensor], dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def forward_single(self, graph, global_features):
        node_features, edge_index = graph.x, graph.edge_index
        x = node_features
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Repeat each tensor value based on the number of nodes in each graph
        num_nodes_per_graph = node_features.shape[0]
        repeated_tensor = global_features.repeat(num_nodes_per_graph, 1)

        # Concatenate the repeated tensor with node features
        x = torch.cat([x, repeated_tensor], dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)