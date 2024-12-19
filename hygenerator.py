
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))
    
    
class AdaIN(nn.Module):
    def __init__(self, noise_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(noise_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, noise_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, noise_dim)

    def _build_weights(self, dim_in, dim_out, noise_dim=64):
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(noise_dim, dim_in)
        self.norm2 = AdaIN(noise_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    
    
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv1d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        return x 
    
    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x 

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance
    
        
class HyGenerator(nn.Module):
    def __init__(self, args, img_size=256, noise_dim=64, max_conv_dim=512, w_hpf=1, device=1, size_dist=False):
        super().__init__()    
        dim_in = 2**14 // img_size
        self.img_size = img_size
        
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        
        self.from_rgb = nn.Conv1d(1, dim_in, 3, 1, 1)
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm1d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim_in, 1, 1, 1))
        
        self.first_layer = nn.Sequential(
            nn.Linear(args.nv, args.dim_edge),
            nn.BatchNorm1d(args.dim_edge, 0.8),
            nn.LeakyReLU(0.2))
        
        if args.dim_edge == 400:       
            self.last_layer = nn.Sequential(
                nn.Linear(384, args.nv),
                nn.BatchNorm1d(args.nv, 0.8),
                nn.LeakyReLU(0.2))
        
        elif args.dim_edge == 600:       
            self.last_layer = nn.Sequential(
                nn.Linear(576, args.nv),
                nn.BatchNorm1d(args.nv, 0.8),
                nn.LeakyReLU(0.2))      
            
        else:            
            self.last_layer = nn.Sequential(
                nn.Linear(args.dim_edge, args.nv),
                nn.BatchNorm1d(args.nv, 0.8),
                nn.LeakyReLU(0.2))
        
        self.size_dist = size_dist
        self.device = device
        
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, noise_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, noise_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)
            
    def forward(self, args, bs, pos_hedges, pos_hedges_feat, z=None):
        indices = []
        neg_samples_onehot = []
        num_neg = len(pos_hedges)
        if not self.size_dist :
            sampled_size = [5] * bs
        else :
            vals = list(self.size_dist.keys())
            p = [self.size_dist[v] for v in vals]
            sampled_size = np.random.choice(vals, bs, p=p)
                 
        num_node = args.nv
        pos_samples_onehot = []
        
        for sublist in pos_hedges:
            onehot_vector = [0] * num_node 
            for idx in sublist:
                onehot_vector[idx - 1] = 1  
            pos_samples_onehot.append(onehot_vector)
            
        x = torch.tensor(pos_samples_onehot, dtype=torch.float).to(self.device)
        s = torch.rand(num_neg, args.noise_dim, device=self.device) 
        
        x = self.first_layer(x).unsqueeze(1)
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        x = self.to_rgb(x).squeeze(1)
        gen_hedge = self.last_layer(x)
        
        indices = []
        neg_samples_onehot = []
        for neg_i in range(num_neg):
            onehots = torch.zeros(sampled_size[neg_i], args.nv).to(self.device)
            values, idx = torch.topk(gen_hedge[neg_i].squeeze(), k=sampled_size[neg_i])
            for i in range(sampled_size[neg_i]):
                onehots[i, idx[i]] = 1 + values[i] - values[i].detach()
            neg_samples_onehot.append(onehots)
            del onehots
            indices.append(idx.detach().to('cpu'))
            
        return pos_samples_onehot, neg_samples_onehot, indices
