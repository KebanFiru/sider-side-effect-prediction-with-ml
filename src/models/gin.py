import torch
from torch_geometric.nn import MLP, GINEConv, GlobalAttention, global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F
from src.data.featurize import NUM_ATOM_FEATURES, NUM_BOND_FEATURES, NUM_GLOBAL_FEATURES, smiles_to_graph

READOUTS = {
    "add": global_add_pool,
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}

class GINEncoder(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, out_layers, dropout=0.2, readout="mean",
                 num_global_features=NUM_GLOBAL_FEATURES, task_attention=False):
        super().__init__()

        self.dropout = dropout
        self.task_attention = task_attention
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINEConv(nn=mlp, train_eps=True, edge_dim = NUM_BOND_FEATURES))
            in_channels = hidden_channels

        if task_attention:
            self.task_attentions = torch.nn.ModuleList([
                GlobalAttention(gate_nn=torch.nn.Linear(hidden_channels, 1))
                for _ in range(out_layers)
            ])
            self.task_heads = torch.nn.ModuleList([
                MLP([hidden_channels + num_global_features, hidden_channels, 1], norm=None, dropout=dropout)
                for _ in range(out_layers)
            ])
        else:
            self.readout = READOUTS[readout]
            self.mlp = MLP([hidden_channels + num_global_features, hidden_channels, out_layers],
                    norm=None, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, batch, global_features):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task_attention:
            outs = []
            for attention, head in zip(self.task_attentions, self.task_heads):
                pooled = attention(x, batch)
                combined = torch.cat([pooled, global_features], dim=1)
                outs.append(head(combined))
            return torch.cat(outs, dim=1)

        pooled = self.readout(x, batch)
        combined = torch.cat([pooled, global_features], dim=1)
        return self.mlp(combined)

if __name__ == "__main__":
    model = GINEncoder(in_channels=NUM_ATOM_FEATURES, hidden_channels=128, num_layers=4, out_layers=27,
                        task_attention=True)
    data = smiles_to_graph("CCO")
    out = model(data.x, data.edge_index, data.edge_attr, torch.zeros(data.x.size(0), dtype=torch.long), data.global_features)
    print(out.shape)
