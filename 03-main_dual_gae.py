# Modified version of main training script to train two GAEs separately: one on ASD, one on control subjects

# Mse - No HOFC - Sepc loss
import os
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_dense_adj
from sklearn.decomposition import PCA


from imports.ABIDEDataset import ABIDEDataset
from imports.network import (WeightedGraphSAGEEncoder, MLPDecoder, decorrelation_loss, get_hard_negatives)
from imports.classifiers import MLPClassifier, get_knn, get_svm, get_nb, get_log_reg, get_rf, train_and_select_classifier
from imports.utils import train_val_test_split_label, augment_graph_with_noise, plot_learning_curves
from imports.explainability import (print_graph_explanation_report,
                                    visualize_graph_explanation,
                                    plot_influential_rois_for_graph)

torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
parser.add_argument("--classifier", type=str, default="mlp", choices=["mlp", "knn", "svm", "nb", "logreg", "rf"])
parser.add_argument('--dataroot', type=str, default='/content/drive/MyDrive/abide_dataset/ABIDE_pcp/ccs/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--hidden', type=int, default=96, help='feature dim')
parser.add_argument('--indim', type=int, default=200, help='feature dim')
parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--load_pretrained_gae', action='store_true', help='Load pre-trained CTL and ASD GAEs and freeze them')
parser.add_argument('--ctl_gae_ckpt', type=str, default='', help='Path to CTL GAE checkpoint')
parser.add_argument('--asd_gae_ckpt', type=str, default='', help='Path to ASD GAE checkpoint')

opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
name = 'ABIDE'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
hidden = opt.hidden
fold = opt.fold

print("----- START File: {}: Parameters: fold={}, num_epoch={}, hidden_size={}, batch_size={} -----".format(__file__, fold, num_epoch, hidden, opt.batchSize))

################## Define Dataloader ##################################

dataset_raw = ABIDEDataset(path, name)
dataset_raw.data.y = dataset_raw.data.y.squeeze()
dataset_raw.data.x[dataset_raw.data.x == float('inf')] = 0

# # Build new dataset with augmentation
from torch_geometric.data import InMemoryDataset

class AugmentedDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        data, slices = self.collate(data_list)
        self.data = data
        self.slices = slices


# Indices of ASD and CTL
asd_indices = (dataset_raw.data.y == 1).nonzero(as_tuple=True)[0].tolist()
ctl_indices = (dataset_raw.data.y == 0).nonzero(as_tuple=True)[0].tolist()

# ASD splits
asd_train_idx, asd_val_idx, asd_test_idx = train_val_test_split_label(asd_indices, kfold=10, fold=fold)

# CTL splits
ctl_train_idx, ctl_val_idx, ctl_test_idx = train_val_test_split_label(ctl_indices, kfold=10, fold=fold)

# --- THEN: augment ONLY the training graphs ---
train_indices = asd_train_idx + ctl_train_idx

augmented_list = []
orig_to_aug = defaultdict(list)  # maps orig index → list of new augmented indices

current_idx = 0
for i in range(len(dataset_raw)):
    orig = dataset_raw[i]
    augmented_list.append(orig)

    # record original index mapping
    orig_to_aug[i].append(current_idx)
    current_idx += 1

    # add augmentations only for training indices
    if i in train_indices:
        noisy_versions = augment_graph_with_noise(orig, num_aug=1, noise_std=0.015)
        for g in noisy_versions:
            augmented_list.append(g)
            orig_to_aug[i].append(current_idx)
            current_idx += 1

dataset = AugmentedDataset(augmented_list)
print("Dataset augmented from", len(dataset)/2, "subjects ->", len(dataset), "graphs total.")

def expand_indices(original_indices):
    new_list = []
    for idx in original_indices:
        new_list.extend(orig_to_aug[idx])  # original + augmented if any
    return new_list

asd_train_aug = expand_indices(asd_train_idx)
asd_val_aug   = expand_indices(asd_val_idx)
asd_test_aug  = expand_indices(asd_test_idx)

ctl_train_aug = expand_indices(ctl_train_idx)
ctl_val_aug   = expand_indices(ctl_val_idx)
ctl_test_aug  = expand_indices(ctl_test_idx)


# Create datasets
asd_train_set = dataset[asd_train_aug]
asd_val_set   = dataset[asd_val_aug]
asd_test_set  = dataset[asd_test_aug]

ctl_train_set = dataset[ctl_train_aug]
ctl_val_set   = dataset[ctl_val_aug]
ctl_test_set  = dataset[ctl_test_aug]

# Combine for loaders
asd_train_loader = DataLoader(asd_train_set, batch_size=opt.batchSize, shuffle=True)
asd_val_loader   = DataLoader(asd_val_set, batch_size=opt.batchSize, shuffle=False)
#asd_test_loader  = DataLoader(asd_test_set, batch_size=opt.batchSize, shuffle=False)

ctl_train_loader = DataLoader(ctl_train_set, batch_size=opt.batchSize, shuffle=True)
ctl_val_loader   = DataLoader(ctl_val_set, batch_size=opt.batchSize, shuffle=False)
#ctl_test_loader  = DataLoader(ctl_test_set, batch_size=opt.batchSize, shuffle=False)

train_set = asd_train_set + asd_val_set + ctl_train_set + ctl_val_set

# For testing
test_set = asd_test_set + ctl_test_set

# NEW: Split test_set into classifier train and test subsets
clf_train_set, clf_test_set = train_test_split(test_set, test_size=0.2, stratify=[g.y.item() for g in test_set], random_state=42)

classifier_train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
classifier_test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


print("~~~~~~~ DATASET STATS ~~~~~~~")
print("Num of ASD subjects", len(asd_indices))
print("Num of CTL subjects", len(ctl_indices))
print("Size of train set",len(classifier_train_loader))
print("Size of test set",len(classifier_test_loader))
print("Size of ASD in test set",len(asd_test_set))
print("Size of CTL in test set",len(ctl_test_set))

print(" ~~~~~~ EXAMPLE OF A SUBJECT ~~~~~~")
print("Subject 5: ", (dataset[4]))
print("Subject 5 x: ", (dataset[4].x))
print("Subject 5 edge attr: ", (dataset[4].edge_attr))

print(" ~~~~~~ EXAMPLE OF A SUBJECT B~~~~~~")
print("Subject 15: ", (dataset[14]))
print("Subject 15 edge attr: ", (dataset[14].edge_attr))

# Define model trainer for GAE
def train_gae(train_loader, val_loader, opposite_loader, label, save_path,
              lr=1e-4, weight_decay=1e-5,hidden=hidden):
    """
    Train one class-specific GAE:
    - train_loader: graphs belonging to THIS class (ASD or CTL)
    """

    encoder = WeightedGraphSAGEEncoder(in_channels=opt.indim, hidden_channels=hidden).to(device)
    decoder = MLPDecoder(hidden_dim=hidden).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=lr, weight_decay=weight_decay
    )
    
     # iterator for opposite class graphs
    opposite_iter = iter(opposite_loader)
    
    # specialization hyperparameters
    margin = 0.02
    alpha = 1   # weight of specialization term (tune 0.5–2.0)

    best_val = float('inf')

    for epoch in range(num_epoch):

        encoder.train(); decoder.train()
        train_loss = 0.0
        # accumulators at epoch start
        recon_mse_sum = 0.0
        opp_mse_sum   = 0.0
        n_batches = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # === SAME-CLASS RECONSTRUCTION (TARGET) ===
            z = encoder(data.x, data.edge_index, data.edge_attr)
            full_edges = list(FULL_EDGE_MAP.keys())
            full_edge_index = torch.tensor(full_edges).t().contiguous().to(device)
            recon = decoder(z, full_edge_index).squeeze()
            target= generate_undirected_edge_attrs(NUM_EDGES, data).squeeze()   
            
            same_mse  = F.mse_loss(recon, target)

            
            # -------------------------------
            # OPPOSITE-CLASS RECONSTRUCTION (SHOULD BE BAD)
            # -------------------------------
            try:
                data_op = next(opposite_iter)
            except StopIteration:
                opposite_iter = iter(opposite_loader)
                data_op = next(opposite_iter)

            data_op = data_op.to(device)

            z_op = encoder(data_op.x, data_op.edge_index, data_op.edge_attr)
            recon_op = decoder(z_op, full_edge_index).squeeze()

            target_op = generate_undirected_edge_attrs(NUM_EDGES, data_op).squeeze()  

            opp_mse = F.mse_loss(recon_op, target_op)
            spec_loss = alpha * torch.clamp(margin - opp_mse, min=0.0)

            # -------------------------------
            # TOTAL LOSS
            # -------------------------------
            loss = same_mse + spec_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch:03d} | {label} Train Loss: {avg_train_loss:.6f}")

        # ---------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------
        encoder.eval(); decoder.eval()
        val_loss = 0.0
        opposite_iter_val = iter(opposite_loader)
        
        with torch.no_grad():

            for data in val_loader:
                data = data.to(device)

                # ===== SAME-CLASS RECONSTRUCTION =====
                z = encoder(data.x, data.edge_index, data.edge_attr)
                full_edges = list(FULL_EDGE_MAP.keys())
                full_edge_index = torch.tensor(full_edges).t().contiguous().to(device)
                recon = decoder(z, full_edge_index).squeeze()

                target = generate_undirected_edge_attrs(NUM_EDGES, data).squeeze()  

                same_mse = F.mse_loss(recon, target)
                
                # ===== OPPOSITE-CLASS RECONSTRUCTION =====
                try:
                    data_op = next(opposite_iter_val)
                except StopIteration:
                    opposite_iter_val = iter(opposite_loader)
                    data_op = next(opposite_iter_val)

                data_op = data_op.to(device)
                z_op = encoder(data_op.x, data_op.edge_index, data_op.edge_attr)
                recon_op = decoder(z_op, full_edge_index).squeeze()
                target_op = generate_undirected_edge_attrs(NUM_EDGES, data_op).squeeze()  
                opp_mse = F.mse_loss(recon_op, target_op)

                # ===== TOTAL VALIDATION LOSS =====
                spec_val = alpha * torch.clamp(margin - opp_mse, min=0.0)
                total_val = same_mse + spec_val
                val_loss += total_val.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:03d} | {label} Val Loss: {avg_val_loss:.6f}")

        if val_loss < best_val:
            print("Saved best model based on validation loss.")
            best_val = val_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
            }, os.path.join(save_path, f"{label}_GAE.pth"))

    # load best
    state = torch.load(os.path.join(save_path, f"{label}_GAE.pth"))
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    return encoder, decoder




# --------- Prepare Features for Classifier ---------
def get_full_edge_map(num_nodes=111):
    edge_map = {}
    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_map[(i, j)] = idx
            idx += 1
    return edge_map  # maps (u, v) to a unique feature index

FULL_EDGE_MAP = get_full_edge_map()
NUM_EDGES = len(FULL_EDGE_MAP)  # should be 6105
print("NUM_EDGES",NUM_EDGES)

def compute_delta_e(recon_asd, recon_ctl, edge_attr):
    """
    Computes ΔE (difference of reconstruction errors) for a graph between ASD and Control GAEs.
    
    Parameters:
        recon_asd (Tensor): Reconstructed adjacency matrix by ASD-GAE. Shape: [nroi, nroi]
        recon_ctl (Tensor): Reconstructed adjacency matrix by Control-GAE. Shape: [nroi, nroi]
        edge_attr (Tensor): Edge weights of the original graph (the ground truth). Shape: [num_edges]
        device (torch.device): CUDA/CPU device
        
    Returns:
        delta_e_vector (Tensor): Per-edge ΔE vector corresponding to FULL_EDGE_MAP indexing
        edge_to_delta (dict): Mapping {(i, j): ΔE} for all edges in the FULL_EDGE_MAP
    """
    # Compute element-wise reconstruction error w.r.t. true adjacency
    err_asd = F.mse_loss(recon_asd, edge_attr, reduction='none')
    err_ctl = F.mse_loss(recon_ctl, edge_attr, reduction='none')
    
    # ΔE: element-wise difference
    delta_e = (err_asd - err_ctl).detach().cpu()  # Shape: [nroi, nroi]
    
    return delta_e

def generate_undirected_edge_attrs(NUM_EDGES, data):
    # STEP 1: Convert directed edge_attr to undirected using FULL_EDGE_MAP
    edge_index_cpu = data.edge_index.cpu().t().tolist()
    edge_attr = data.edge_attr.cpu()

    undirected_edge_attr = torch.zeros((NUM_EDGES, edge_attr.size(1)))
    edge_count = torch.zeros(NUM_EDGES)

    for idx, (u, v) in enumerate(edge_index_cpu):
        i, j = min(u, v), max(u, v)
        if (i, j) in FULL_EDGE_MAP:
            eid = FULL_EDGE_MAP[(i, j)]
            undirected_edge_attr[eid] += edge_attr[idx]
            edge_count[eid] += 1

    # Avoid division by zero
    edge_count[edge_count == 0] = 1
    undirected_edge_attr = undirected_edge_attr / edge_count.unsqueeze(1)
    undirected_edge_attr = undirected_edge_attr.to(device)

    return undirected_edge_attr


def extract_gae_features(loader, encoder_asd, decoder_asd, encoder_ctl, decoder_ctl,
                          FULL_EDGE_MAP, NUM_EDGES, top_k=10, save_dir=None):
    encoder_asd.eval()
    decoder_asd.eval()
    encoder_ctl.eval()
    decoder_ctl.eval()

    features, labels = [], []
    top_edges_all_graphs, top_nodes_all_graphs = [], []
    print_count = 0

    all_true_edges = []
    all_recon_asd = []
    all_recon_ctl = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # --- Ground-truth edges (undirected) ---
            undirected_edge_attr = generate_undirected_edge_attrs(NUM_EDGES, data).to(device)
            all_true_edges.append(undirected_edge_attr.cpu().numpy())

            # --- Encode with both GAEs ---
            z_asd = encoder_asd(data.x, data.edge_index, data.edge_attr)
            z_ctl = encoder_ctl(data.x, data.edge_index, data.edge_attr)
            z_norm = F.normalize(z_asd, p=2, dim=1)

            # --- Decode all undirected edges ---
            full_edges = list(FULL_EDGE_MAP.keys())
            full_edge_index = torch.tensor(full_edges).t().contiguous().to(device)

            recon_asd = decoder_asd(z_asd, full_edge_index).squeeze()
            recon_ctl = decoder_ctl(z_ctl, full_edge_index).squeeze()
            
            all_recon_asd.append(recon_asd.cpu().numpy())
            all_recon_ctl.append(recon_ctl.cpu().numpy())

            # --- Compute ΔE (difference in reconstruction error) ---
            delta_e = compute_delta_e(recon_asd, recon_ctl, undirected_edge_attr.squeeze())
            delta_e = delta_e / (delta_e.norm(p=2) + 1e-8)   # normalize
            
            # --- Find top-K most influential edges ---
            edge_to_delta = {}
            for (i, j), eid in FULL_EDGE_MAP.items():
                delta = delta_e[eid].item()
                edge_to_delta[(i, j)] = delta
            influential_edges = sorted(edge_to_delta.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            top_edges_all_graphs.append(influential_edges)

            # --- Find Top-K influential nodes ---
            node_scores = defaultdict(float)
            for (u, v), delta in edge_to_delta.items():
                node_scores[u] += abs(delta)
                node_scores[v] += abs(delta)

            top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top_nodes_all_graphs.append(top_nodes)

            # ----------------------------------------
            # STEP 5: Save features/labels
            features.append(delta_e)
            labels.append(data.y.item())

    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels)

    # --- Save all results per fold ---
    np.save(os.path.join(save_dir, f"train_true_edges_fold_{fold}.npy"), np.array(all_true_edges))
    np.save(os.path.join(save_dir, f"train_recon_asd_fold_{fold}.npy"), np.array(all_recon_asd))
    np.save(os.path.join(save_dir, f"train_recon_ctl_fold_{fold}.npy"), np.array(all_recon_ctl))
    np.save(os.path.join(save_dir, f"train_labels_fold_{fold}.npy"), labels_tensor.cpu().numpy())

    print(f"Saved train ground truth edges and reconstructions for fold {fold} to {save_dir}/")

    return features_tensor, labels_tensor, top_edges_all_graphs, top_nodes_all_graphs



# --------- Evaluate Classifier on Test Set ---------
def evaluate_classifier(classifier, test_loader, encoder_asd, decoder_asd, encoder_ctl, decoder_ctl, scaler, selector, save_dir, is_sklearn):
    
    if not is_sklearn:
        classifier.eval()
    y_true, y_pred, confidences = [], [], []

    features, labels = [], []

    all_true_edges = []
    all_recon_asd = []
    all_recon_ctl = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            z_asd = encoder_asd(data.x, data.edge_index, data.edge_attr)
            z_ctl = encoder_ctl(data.x, data.edge_index, data.edge_attr)

            undirected_edge_attr = generate_undirected_edge_attrs(NUM_EDGES, data)

            all_true_edges.append(undirected_edge_attr.cpu().numpy())

            # use full undirected edge list for decoding
            full_edges = list(FULL_EDGE_MAP.keys())
            full_edge_index = torch.tensor(full_edges).t().contiguous().to(device)
            
            recon_asd = decoder_asd(z_asd, full_edge_index).squeeze()
            recon_ctl = decoder_ctl(z_ctl, full_edge_index).squeeze()

            all_recon_asd.append(recon_asd.cpu().numpy())
            all_recon_ctl.append(recon_ctl.cpu().numpy())

            # === Compute Δ(Err) ===
            delta_e = compute_delta_e(recon_asd, recon_ctl, undirected_edge_attr.squeeze())
            
            # === Normalize ΔE vector ===
            delta_e = delta_e / (delta_e.norm(p=2) + 1e-8)

            # === Scale ===
            delta_np = delta_e.unsqueeze(0).numpy()
            
            delta_scaled = scaler.transform(delta_np)
            X_val_selected = selector.transform(delta_scaled)
            
            if is_sklearn:
                # sklearn classifier
                prob = classifier.predict_proba(X_val_selected)[0,1] if hasattr(classifier, "predict_proba") else classifier.decision_function(X_val_selected)[0]
                pred = 1 if prob > 0.5 else 0
            else:
                # PyTorch classifier
                feature_tensor = torch.tensor(X_val_selected, dtype=torch.float32).to(device)
                logit = classifier(feature_tensor)
                prob = torch.sigmoid(logit).item()
                pred = 1 if prob > 0.5 else 0

            y_pred.append(pred)
            y_true.append(data.y.item())
            confidences.append(prob)
            features.append(delta_e)
            labels.append(data.y.item())

    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels)

    # Save combined numpy files for the test set
    np.save(os.path.join(save_dir, f"test_true_edges_fold_{fold}.npy"), np.array(all_true_edges))
    np.save(os.path.join(save_dir, f"test_recon_asd_fold_{fold}.npy"), np.array(all_recon_asd))
    np.save(os.path.join(save_dir, f"test_recon_ctl_fold_{fold}.npy"), np.array(all_recon_ctl))
    np.save(os.path.join(save_dir, f"test_labels_fold_{fold}.npy"), np.array(labels))
    print(f"Saved test ground truth edges and reconstructions for fold {fold} to {save_dir}/")
    
    # === Evaluation metrics ===
    acc    = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1‑score : {f1:.4f}\n")
    print(f"Confusion Matrix:\n{cm}\n")

    return y_true, y_pred, confidences, features_tensor, labels_tensor


# Step4: Training both GAEs or Loading them
save_dir = "./dual_model_checkpoints"
os.makedirs(save_dir, exist_ok=True)


if opt.load_pretrained_gae:
    # Initialize models and load checkpoints
    asd_encoder = WeightedGraphSAGEEncoder(in_channels=opt.indim, hidden_channels=hidden).to(device)
    asd_decoder = MLPDecoder(hidden_dim=hidden).to(device)
    ctl_encoder = WeightedGraphSAGEEncoder(in_channels=opt.indim, hidden_channels=hidden).to(device)
    ctl_decoder = MLPDecoder(hidden_dim=hidden).to(device)
  
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.asd_gae_ckpt:
        asd_ckpt = torch.load(opt.asd_gae_ckpt, map_location=device)
        asd_encoder.load_state_dict(asd_ckpt['encoder'])
        asd_decoder.load_state_dict(asd_ckpt['decoder'])
        asd_classifier.load_state_dict(asd_ckpt['classifier'])
        for p in asd_encoder.parameters():
            p.requires_grad = False
        for p in asd_decoder.parameters():
            p.requires_grad = False
        for p in asd_classifier.parameters():
            p.requires_grad = False
        print("Loaded and froze ASD GAE.")

    if opt.ctl_gae_ckpt:
        ctl_ckpt = torch.load(opt.ctl_gae_ckpt, map_location=device)
        ctl_encoder.load_state_dict(ctl_ckpt['encoder'])
        ctl_decoder.load_state_dict(ctl_ckpt['decoder'])
        ctl_classifier.load_state_dict(ctl_ckpt['classifier'])
        for p in ctl_encoder.parameters():
            p.requires_grad = False
        for p in ctl_decoder.parameters():
            p.requires_grad = False
        for p in ctl_classifier.parameters():
            p.requires_grad = False
        
        print("Loaded and froze CTL GAE.")
else:
    # Train both GAEs from scratch
    print("\nSTART TRAINING DUAL GAE!")
    asd_encoder, asd_decoder = train_gae(asd_train_loader, asd_val_loader, ctl_val_loader, "ASD", save_dir)
    ctl_encoder, ctl_decoder = train_gae(ctl_train_loader, ctl_val_loader, asd_val_loader, 'CTL', save_dir)



# Step 5: Train classifier
train_features_raw, train_labels, top_edges_all_graphs, top_nodes_all_graphs = extract_gae_features(classifier_train_loader, asd_encoder, asd_decoder, ctl_encoder, ctl_decoder, FULL_EDGE_MAP, NUM_EDGES, top_k=10, save_dir='data/edge_weights')
print("train_features_raw shape",train_features_raw.shape)

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features_np = train_features_raw.cpu().numpy()
train_labels_np = train_labels.cpu().numpy()
train_features_scaled = scaler.fit_transform(train_features_np)

# Select the 300 best features because 6105 are too many
# Could use PCA instead of SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=300)  # or try 200
train_features_selected = selector.fit_transform(train_features_scaled, train_labels)
train_features_tensor = torch.tensor(train_features_selected).float().to(device)
print("train_features_tensor shape",train_features_tensor.shape)

# Create and train classifier
clf = train_and_select_classifier(opt, train_features_tensor, train_labels, device)

# Debug
def check_gae_specialization(test_loader, encoder_asd, decoder_asd, encoder_ctl, decoder_ctl):
    """
    Verifies the core assumption:
    ASD test graph -> ASD-GAE should reconstruct better
    CTL test graph -> CTL-GAE should reconstruct better
    """
    encoder_asd.eval()
    decoder_asd.eval()
    encoder_ctl.eval()
    decoder_ctl.eval()

    correct = 0
    total = 0

    print("\n========== CHECKING GAE SPECIALIZATION ==========\n")

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)

            # Get full undirected target adjacency (shape = [6105, 1])
            target = generate_undirected_edge_attrs(NUM_EDGES, data).squeeze()

            # Encode
            z_asd = encoder_asd(data.x, data.edge_index, data.edge_attr)
            z_ctl = encoder_ctl(data.x, data.edge_index, data.edge_attr)

            # Decode full adjacency
            full_edges = list(FULL_EDGE_MAP.keys())
            full_edge_index = torch.tensor(full_edges).t().contiguous().to(device)

            recon_asd = decoder_asd(z_asd, full_edge_index).squeeze()
            recon_ctl = decoder_ctl(z_ctl, full_edge_index).squeeze()

            # Compute REAL reconstruction error
            err_asd = F.mse_loss(recon_asd, target, reduction='mean').item()
            err_ctl = F.mse_loss(recon_ctl, target, reduction='mean').item()

            y = data.y.item()   # 1 = ASD, 0 = CTL

            # Prediction based purely on which GAE fits best
            pred = 1 if err_asd < err_ctl else 0

            correct += (pred == y)
            total += 1

            print(f"[Graph {i}] true={y}, err_ASD={err_asd:.4f}, err_CTL={err_ctl:.4f}, pred={pred}")

    acc = correct / total
    print(f"\nGAE Specialization Accuracy = {acc:.4f}\n")

    return acc

#test_decoder_difference(classifier_test_loader, asd_encoder, asd_decoder, ctl_encoder, ctl_decoder)
check_gae_specialization(classifier_test_loader, 
                         asd_encoder, asd_decoder, 
                         ctl_encoder, ctl_decoder)

# Step 6: Evaluate on test set
is_sklearn = opt.classifier in ["knn", "svm", "nb", "logreg", "rf"]
y_true, y_pred, confidences, raw_test_features, test_labels = evaluate_classifier(
    clf, classifier_test_loader,
    asd_encoder, asd_decoder, ctl_encoder, ctl_decoder,
    scaler, selector, save_dir='data/edge_weights', is_sklearn=is_sklearn
)
test_features_np = raw_test_features.cpu().numpy()
test_labels_np = test_labels.cpu().numpy()

# === SAVE DATA FOR PERMUTATION TEST FOR THIS FOLD ===
X_train = train_features_selected              # numpy array already
y_train = train_labels_np                      # numpy

# Transform test features exactly like evaluation
X_test_raw = test_features_np                      # already created in your script
X_test_scaled = scaler.transform(test_features_np)
X_test = selector.transform(X_test_scaled)
y_test = test_labels.cpu().numpy()

np.save(f"perm_data/X_train_fold{fold}.npy", X_train)
np.save(f"perm_data/y_train_fold{fold}.npy", y_train)
np.save(f"perm_data/X_test_fold{fold}.npy", X_test)
np.save(f"perm_data/y_test_fold{fold}.npy", y_test)
print(f"Permutation test data for fold {fold} saved.")

#################################################################
import pickle

with open("data/top_nodes_all_graphs.pkl", "wb") as f:
    pickle.dump(top_nodes_all_graphs, f)

# Suppose you want to inspect graph index 0
graph_idx = 0

# Load the atlas labels
with open("data/harvard_oxford_111_labels.json", "r") as f:
    harvard_oxford_111_labels = json.load(f)

# Get top influential edges and node scores for this graph
edge_list = top_edges_all_graphs[graph_idx]  # List of ((u, v), delta) pairs
node_scores_dict = dict(top_nodes_all_graphs[graph_idx])  # Convert to dict if needed


# Explanation function 1
print_graph_explanation_report(graph_idx, y_true, y_pred, confidences, 
                                  top_edges_all_graphs, top_nodes_all_graphs)

# Explanation function 2
# visualize_graph_explanation(
#     edge_list=edge_list,
#     node_scores=node_scores_dict,
#     top_k_nodes=10,
#     graph_idx=graph_idx
# )

# Explanation function 3
# plot_influential_rois_for_graph(graph_idx, 
#                                 top_nodes_all_graphs, 
#                                 harvard_oxford_111_labels, 
#                                 top_k=12)