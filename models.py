import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
    
class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

class multilayers(nn.Module):
    def __init__(self, model, inputs, n_layers, memory_dim, K=32):
        super(multilayers, self).__init__()
        self.layers = []
        self.model = model
        self.memory_size=K
        if self.memory_size > 0 :
            self.register_buffer("memory", torch.zeros(K, memory_dim))
        for i in range(n_layers):
            self.layers.append(self.model(*inputs))
        for i in range(n_layers):
            self.add_module(f'layer{i}', self.layers[i])
    
    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        if self.memory_size > 0:
            self.memory = self.memory.to(device)
        return self
        
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
            
    def forward(self, inputs, n_layers):
        first_layer, last_layer = True, False
        for i, layer in enumerate(self.layers):
            if i == n_layers-1:
                last_layer = True
                return layer(*inputs, first_layer, last_layer)
            else:
                inputs = layer(*inputs, first_layer, last_layer)
            first_layer = False
            
class HNHN(nn.Module):
    def __init__(self, input_dim, dim_vertex, dim_edge):
        super(HNHN, self).__init__()
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, dim_vertex)
        self.vtx_lin = torch.nn.Linear(dim_vertex, dim_vertex)
        
        self.ve_lin = torch.nn.Linear(dim_vertex, dim_edge)
        self.ev_lin = torch.nn.Linear(dim_edge, dim_vertex)
        
    def weight_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']
        return {'weight': weight}
    
    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'weight': edges.data['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        aggr = torch.sum(Weight * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}

    def forward(self, g, vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum, first_layer, last_layer):
        with g.local_scope():
            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
            else:
                feat_v = self.vtx_lin(vfeat)  
                
            feat_e = efeat

            g.ndata['h'] = {'node': feat_v}
            g.ndata['Wh'] = {'node' : self.ve_lin(feat_v)}
            g.ndata['reg_weight'] = {'node':v_reg_weight, 'edge':e_reg_weight}
            g.ndata['reg_sum'] = {'node':v_reg_sum, 'edge':e_reg_sum}
            
            # edge aggregation
            g.apply_edges(self.weight_fn, etype='in')
            g.update_all(self.message_func, self.reduce_func, etype='in')            
            feat_e = g.ndata['h']['edge']
            
            g.ndata['Wh'] = {'edge' : self.ev_lin(feat_e)}
            
            # node aggregattion
            g.apply_edges(self.weight_fn, etype='con')
            g.update_all(self.message_func, self.reduce_func, etype='con')
            feat_v = g.ndata['h']['node']
            
            if last_layer:
                return feat_v, feat_e
            else:
                return [g, feat_v, feat_e, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum]
    
