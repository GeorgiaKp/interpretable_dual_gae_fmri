import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected, add_self_loops


# Edge-Conditioned SAGEConv Layer
class EdgeConditionedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_feat_dim):
        super().__init__(aggr='add')  # aggregate by summation

        self.root_lin = nn.Linear(in_channels, out_channels)

        # MLP: transforms [x_j | edge_attr] --> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

        self.edge_feat_dim = edge_feat_dim

    def forward(self, x, edge_index, edge_attr):
        """
        x:            [num_nodes, in_channels]
        edge_index:   [2, num_edges]
        edge_attr:    [num_edges, edge_feat_dim]
        """

        # Ensure shape is correct
        edge_attr = edge_attr.view(-1, self.edge_feat_dim)

        num_nodes = x.size(0)

        # Build dummy zero edge_attr for self-loops
        loop_attr = torch.zeros((num_nodes, self.edge_feat_dim),
                                device=edge_attr.device)

        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_attr,
            fill_value=0.0,       # attributes for self-loops → zeros
            num_nodes=num_nodes
        )

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Add skip connection
        out += self.root_lin(x)

        return out

    def message(self, x_j, edge_attr):
        """
        Message from j → i:
        m_ij = MLP([x_j | e_ij])
        """

        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

class EdgeConditionedSAGEEncoder(nn.Module):
    def __init__(self, in_channels, edge_feat_dim,
                 hidden_channels=64, num_layers=3, dropout=0.1):

        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(
            EdgeConditionedSAGEConv(in_channels, hidden_channels, edge_feat_dim)
        )
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.layers.append(
                EdgeConditionedSAGEConv(hidden_channels, hidden_channels, edge_feat_dim)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x   

class EdgeAttrCompressor(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)

# GraphSAGE Encoder for GAE
class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "add" for weighted sum
        self.lin = nn.Linear(in_channels, out_channels)
        self.root_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        else:
            edge_weight = edge_weight.view(-1)  # Flatten to 1D

        # Add self-loops with weight=1.0
        edge_index, edge_weight = self.add_self_loops_with_edge_weight(edge_index, edge_weight, num_nodes=x.size(0))

        # Linear transformation first
        x_trans = self.lin(x)

        # Propagate with weighted messages
        out = self.propagate(edge_index, x=x_trans, edge_weight=edge_weight)

        # Add skip connection
        out += self.root_lin(x)

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    @staticmethod
    def add_self_loops_with_edge_weight(edge_index, edge_weight, num_nodes, fill_value=1.0):
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)  # shape [2, num_nodes]

        edge_index = torch.cat([edge_index, loop_index], dim=1)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1) - num_nodes,), device=edge_index.device)

        loop_weight = torch.full((num_nodes,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        return edge_index, edge_weight


class WeightedGraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.02):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(WeightedSAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.layers.append(WeightedSAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_weight)
            x = norm(x)
            x = nn.LeakyReLU(negative_slope=0.1)(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# MLP Decoder
class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z, edge_index):
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        z_cat = torch.cat([z_i, z_j], dim=1)
        # z_cat = torch.abs(z_i - z_j)

        # Debug print
        #print("z_cat std:", z_cat.std().item(), "mean:", z_cat.mean().item())

        return self.mlp(z_cat)

class GraphClassifier(nn.Module):
    def __init__(self, embedding_dim=96, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # or just 1 for binary classification
        )

    def forward(self, x):
        return self.fc(x)

# --------- MLP Classifier ---------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze()


# De-correlation regularization to prevent embedding collapse
def decorrelation_loss(z):
    z = F.normalize(z, dim=1)
    corr = torch.matmul(z.T, z)
    I = torch.eye(corr.size(0), device=z.device)
    return ((corr - I)**2).mean()

def get_hard_negatives(z, pos_edge_index, num_nodes, num_neg_samples=None):
    """
    Selects hard negative edges based on cosine similarity of node embeddings.
    
    Args:
        z (Tensor): Node embeddings, shape [N, D]
        pos_edge_index (Tensor): Positive edge index, shape [2, E]
        num_nodes (int): Number of nodes in the graph
        num_neg_samples (int, optional): Number of hard negatives to return (default = E)

    Returns:
        neg_edge_index (Tensor): Edge index of hard negatives, shape [2, K]
    """
    if num_neg_samples is None:
        num_neg_samples = pos_edge_index.size(1)

    # Ensure undirected, deduplicated edges
    pos_edge_index = to_undirected(pos_edge_index, num_nodes=num_nodes)

    # Normalize embeddings for cosine similarity
    z_norm = F.normalize(z, p=2, dim=1)
    sim_matrix = torch.matmul(z_norm, z_norm.T)  # [N x N]

    # Mask out positive edges and self-loops
    adj = torch.zeros((num_nodes, num_nodes), device=z.device)
    adj[pos_edge_index[0], pos_edge_index[1]] = 1
    adj.fill_diagonal_(1)  # exclude self-loops

    neg_mask = (adj == 0)

    # Flattened similarities for non-edges
    sim_flat = sim_matrix[neg_mask]

    # Select top-K hardest non-edges
    topk_vals, topk_idx = torch.topk(sim_flat, num_neg_samples)

    # Map back to (i, j) format
    neg_indices = neg_mask.nonzero(as_tuple=False)[topk_idx]
    neg_edge_index = neg_indices.t()  # shape [2, num_neg_samples]

    return neg_edge_index