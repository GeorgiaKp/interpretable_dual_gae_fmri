from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def train_val_test_split(n_sub, kfold = 5, fold = 0):
    # n_sub = 1035
    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id
    
def train_val_test_split_label(indices, kfold = 10, fold = 0):
    
    indices = np.array(indices)

    import random
    random.seed(123)
    random.shuffle(indices)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(indices):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = indices[train_index[fold]].tolist()
    test_id = indices[test_index[fold]].tolist()
    val_id = indices[val_index[fold]].tolist()

    return train_id,val_id,test_id


def augment_graph_with_noise(data, num_aug=5, noise_std=0.02):
    augmented = []
    for _ in range(num_aug):
        new_data = data.clone()

        # Split FC / HOFC
        fc = new_data.edge_attr[:,0]
        #hofc = new_data.edge_attr[:,1]

        noise = torch.randn_like(fc) * noise_std
        fc_noisy = fc + noise

        # Recombine
        # new_data.edge_attr = torch.stack([fc_noisy, hofc], dim=1)
        new_data.edge_attr = fc_noisy.unsqueeze(1)

        augmented.append(new_data)

    return augmented

def plot_learning_curves(train_losses, val_losses, label, fold, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{label} GAE Learning Curve, fold = {fold}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Force integer ticks on x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    # Save the figure
    curve_path = os.path.join(save_path, f"{label}_GAE_learning_curve_fold_{fold}.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()

    print(f"Learning curve saved to {curve_path}")


def build_adjacency_matrix(edge_index, edge_attr, num_nodes):
    # Create a dense matrix of zeros on the same device as edge_index.
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    # Populate the matrix with edge attributes.
    # Assuming edge_attr is a column vector (shape [E, 1]), we squeeze to get shape [E].
    adj[edge_index[0], edge_index[1]] = edge_attr.squeeze()
    return adj

def adjacency_to_edge_index(adj, threshold=1e-5):
    # Find indices where the adjacency matrix has values above a small threshold.
    # nonzero returns a tensor of shape [E, 2] (each row is [i, j] for an edge).
    edge_index = (adj > threshold).nonzero(as_tuple=False).t()
    
    # Extract corresponding edge attributes.
    edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1)  # make sure it has shape [E, 1]
    return edge_index, edge_attr

