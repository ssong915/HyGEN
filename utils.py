'''
Utilities functions for the framework.
'''
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from torchmetrics import AveragePrecision
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    
    ##### training hyperparameter #####
    parser.add_argument("--dataset_name", type=str, default='cora', help='dataset name: cora, coraA, citeseer, pubmed, dblp, dblpA')
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=True, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--random", default=0, type=int, help='random seed')
    parser.add_argument("--gpu", type=int, default=0, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--exp_num", default=5, type=int, help='number of experiments')
    parser.add_argument("--epochs", default=200, type=int, help='number of epochs')
    parser.add_argument("--bs", default=128, type=int, help='batch size') 
    parser.add_argument("--neg_sample", default='cns', type=str, help='negative sampling method')
    parser.add_argument("--train_DG", default="epoch1:1", type=str, help='update ratio in epochs (D updates:G updates)')
    parser.add_argument("--testns", type=str, default='SMCA', help='test negative sampler')
    parser.add_argument("--clip", type=float, default='0', help='weight clipping')
    parser.add_argument("--training", type=str, default='wgan', help='loss objective: wgan, none')
    parser.add_argument("--D_lr", default=0.0001, type=float, help='learning rate') 
    parser.add_argument("--G_lr", default=0.0005, type=float, help='learning rate')
    parser.add_argument("--aggregator", default='maxmin', type=str, help='hyperedge feature aggregation : maxmin, average') 
    
    ##### Discriminator architecture #####
    parser.add_argument("--model", default='hnhn', type=str, help='discriminator: hnhn, hgnn')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers') 
    parser.add_argument("--alpha_e", default=0, type=float, help='normalization term for hnhn')
    parser.add_argument("--alpha_v", default=1, type=float, help='normalization term for hnhn')
    parser.add_argument("--dim_hidden", default=64, type=int, help='dimension of hidden vector') 
    parser.add_argument("--dim_vertex", default=64, type=int, help='dimension of vertex hidden vector') 
    parser.add_argument("--dim_edge", default=64, type=int, help='dimension of edge hidden vec') 
    parser.add_argument("--memory_size", default=32, type=int, help='memory size') 
    parser.add_argument("--memory_type", default='sample', type=str)
    parser.add_argument("--noise_dim", default=64, type=int, help='dimension of noise vector')

    ##### Generator architecture #####
    parser.add_argument("--gen", type=str, default='AE', help='generator: encoder-decoder structure')
    
    ##### Additional regularization term ##### 
    parser.add_argument("--loss_used", type=str, default='used', help="Additional term option")
    parser.add_argument("--k", type=float, default=0.2, help="Specific similarity used in regularization term")
    parser.add_argument("--p_values", type=int, default=1, help="Curvature of the regularization term")
    
    ##### General Settings #####
    parser.add_argument("--patience", type=int, default=50, help="early stop")
        
    opt = parser.parse_args()
    print(opt.gpu)
    return opt
     
def gen_size_dist(hyperedges):
    size_dist = {}
    for edge in hyperedges:
        leng = len(edge)
        if leng not in size_dist :
            size_dist[leng] = 0
        size_dist[leng] += 1
    if 1 in size_dist:
        del size_dist[1]
    if 2 in size_dist:
        del size_dist[2]
    total = sum(v for k, v in size_dist.items())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    
    return size_dist  

def convergence_with_high_gradient(sim_matrix, k, p, epsilon=1e-6):
    transformed_matrix = ((torch.abs(sim_matrix - k)) / (sim_matrix * (1 - sim_matrix) + epsilon)) ** p
    return transformed_matrix

def calculate_node_degree(hyperedges):
    if isinstance(hyperedges, list):
        hyperedges = [torch.tensor(edge) if not isinstance(edge, torch.Tensor) else edge for edge in hyperedges]
        hyperedges = torch.stack(hyperedges) 
    hyperedges = hyperedges.numpy()
    node_degrees = np.sum(hyperedges, axis=0)
    return node_degrees
 
def unsqueeze_onehot(onehot):
    edge_size = max(int(onehot.sum().item()), 1)
    onehot_shape = onehot.shape[0]
    unsqueeze = torch.zeros([edge_size, onehot_shape], device=onehot.device)
    nonzero_idx = onehot.nonzero()
    for i, idx in enumerate(nonzero_idx) :
        unsqueeze[i][idx]=1
    return unsqueeze 

def measure(label, pred):
    average_precision = AveragePrecision(task='binary')
    auc_roc = metrics.roc_auc_score(np.array(label), np.array(pred))

    label = torch.tensor(label)
    label = label.type(torch.int64)

    ap = average_precision(torch.tensor(pred), torch.tensor(label))
    return auc_roc, ap

def nt_xent(args, out1, out2, temperature=0.1, distributed=False, normalize=False, epsilon=1e-10):
    assert out1.size(0) == out2.size(0)
    if normalize:
        out1 = F.normalize(out1)
        out2 = F.normalize(out2)
    N = out1.size(0)

    _out = [out1, out2]
    outputs = torch.cat(_out, dim=0)

    sim_matrix = outputs @ outputs.t()    

    if args.loss_used == 'used':
        sim_matrix = sim_matrix / temperature

        sim_matrix = F.softmax(sim_matrix, dim=1)
        transformed_matrix = convergence_with_high_gradient(sim_matrix, args.k, args.p_values)

        loss = torch.sum(transformed_matrix[:N, N:].diag() + transformed_matrix[N:, :N].diag()) / (2*N)

        return loss