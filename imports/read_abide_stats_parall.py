import os.path as osp
from os import listdir
import os
import glob
import h5py

import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_array
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
from imports.gdc import GDC
#from gdc import GDC

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    print("batch.shape:", batch.shape)
    print("batch.min():", batch.min())
    print("batch.max():", batch.max())
    print("row:", row)
    print("batch[row]:", batch[row])

    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# === Embedding loader ===
def load_embedding(subject_id, emb_dir='./ABIDE_pcp/ccs/filt_noglobal/roi_embeddings_normalized_datawide', format='npy'):
    if format == 'npy':
        path = osp.join(emb_dir, f'{subject_id}.npy')
        return np.load(path)
    elif format == 'pt':
        path = osp.join(emb_dir, f'{subject_id}.pt')
        return torch.load(path).numpy()
    else:
        raise ValueError("Unsupported format: use 'npy' or 'pt'")

# === Main data loader ===
# def read_data(data_dir, emb_dir='./ABIDE_pcp/ccs/filt_noglobal/roi_embeddings_normalized_datawide', emb_format='npy'):
#     onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
#     onlyfiles.sort()

#     batch = []
#     pseudo = []
#     y_list = []
#     edge_att_list, edge_index_list, att_list = [], [], []

#     # parallel processing setup
#     cores = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=cores)
#     func = partial(read_sigle_data, data_dir)  # assumes you have this defined elsewhere

#     import timeit
#     start = timeit.default_timer()
#     res = pool.map(func, onlyfiles)
#     pool.close()
#     pool.join()
#     stop = timeit.default_timer()
#     print('Time: ', stop - start)

#     print("len res",len(res))
#     for j in range(len(res)):
#         subject_id = osp.splitext(onlyfiles[j])[0]  # remove '.npz' or other extension
#         try:
#             embedding = load_embedding(subject_id, emb_dir=emb_dir, format=emb_format)
#         except FileNotFoundError:
#             print(f"❌ Missing embedding for {subject_id}, skipping")
#             continue

#         edge_att_list.append(res[j][0])                    # edge attributes
#         edge_index_list.append(res[j][1] + j * res[j][4])  # edge index (offset for batching)
#         att_list.append(embedding)                         # ⬅ our x features
#         y_list.append(res[j][3])                           # label
#         batch.append([j] * res[j][4])                      # node-to-graph assignment
#         pseudo.append(np.diag(np.ones(res[j][4])))         # position info (optional)

#     # stack everything
#     edge_att_arr = np.concatenate(edge_att_list)
#     edge_index_arr = np.concatenate(edge_index_list, axis=1)
#     att_arr = np.concatenate(att_list, axis=0)  # this is x
#     pseudo_arr = np.concatenate(pseudo, axis=0)
#     y_arr = np.stack(y_list)

#     # convert to torch tensors
#     edge_att_torch = torch.tensor(edge_att_arr, dtype=torch.float32).reshape(-1, 1)
#     att_torch = torch.tensor(att_arr, dtype=torch.float32)
#     y_torch = torch.tensor(y_arr, dtype=torch.long)
#     batch_torch = torch.tensor(np.hstack(batch), dtype=torch.long)
#     edge_index_torch = torch.tensor(edge_index_arr, dtype=torch.long)
#     pseudo_torch = torch.tensor(pseudo_arr, dtype=torch.float32)

#     data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch,
#                 edge_attr=edge_att_torch, pos=pseudo_torch)

#     data, slices = split(data, batch_torch)  # assuming split is implemented
#     return data, slices

# === Main data loader, no timeseries embeddings ===
def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)
    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices


def read_sigle_data(data_dir,filename,use_gdc =False):

    temp = dd.io.load(osp.join(data_dir, filename))

    # read edge and edge attribute
    pcorr = temp['pcorr'][()]

    num_nodes = pcorr.shape[0]
    G = from_numpy_array(pcorr)
    A = nx.to_scipy_sparse_array(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp['corr'][()]
    label = temp['label'][()]

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes

if __name__ == "__main__":
    # data_dir = '/home/azureuser/projects/BrainGNN/data/ABIDE_pcp/cpac/filt_noglobal/raw'
    data_dir = '/content/drive/MyDrive/abide_dataset/ccs/filt_noglobal/raw'
    filename = '50003.h5'
    read_sigle_data(data_dir, filename)






