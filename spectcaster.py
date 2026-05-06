import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
import torch.fft
#from timm import DropPath

class RevIN(nn.Module):
    def __init__(self,num_features, eps=  1e-5,affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()
    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    

    def _get_statistics(self,x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self,x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    def _denormalize(self,x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        if self.subtract_last:
            x = x + self.last
        else:
            x = x * self.stdev + self.mean
        return x





class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpectralGatingNetwork(nn.Module):
    def __init__(self,dim,h=14,w=8):
        super(SpectralGatingNetwork, self).__init__()
        self.embed_dim = dim
        self.complex_weight = nn.Parameter(torch.randn(1,self.embed_dim,dtype=torch.float32)*0.02)
        self.bypass = False
        self.scale = 0.02

    def set_bypass(self, bypass):
        self.bypass = bypass

    def reset_parameters(self):
        nn.init.normal_(self.complex_weight, mean=0.0, std=self.scale)

    def forward(self, x):
        if self.bypass:
            return x
        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=1,norm='ortho')
        weight = torch.fft.rfft(self.complex_weight, dim=1,norm='ortho')
        x = x.permute(0,2,1)
        y = x*weight
        out = torch.fft.irfft(y, n = self.embed_dim,norm='ortho')
        out = out.permute(0,2,1)
        return out

class MLP(nn.Module):
    def __init__(self,in_features, hidden_features=None, out_features=None,act_layer = nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x) 
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,dim, mlp_ratio = 4., drop = 0., drop_path=0., act_layer = nn.GELU, norm_layer = nn.BatchNorm1d, h =14, w= 8):
        super(Block, self).__init__()
        self.norm1 = nn.Sequential(Transpose(1,2),norm_layer(dim))
        self.filter = SpectralGatingNetwork(dim,h=h,w=w)
        self.drop_path = nn.Identity()#DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.Sequential(norm_layer(dim))
        self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer = act_layer, drop = drop_path)
    def forward(self, x):
        #(B*C, num_patches, d_model)
        z = self.norm1(x)
       # (B*C, d_model, num_patches)
        z = self.filter(z) 
        #(B*C, d_model, num_patches)
        z = self.norm2(z)
        z = z.permute(0,2,1) # (B*C, num_patches, d_model)
        z = self.mlp(z)
        z = self.drop_path(z)
        x = x + z
        return x

class Block_attention(nn.Module):
    def __init__(self,dim, mlp_ratio=4. ,drop=0., drop_path = 0., act_layer = nn.GELU,norm_layer = nn.BatchNorm1d):
        num_heads = 8 #
        super(Block_attention, self).__init__()
        self.norm1 = nn.Sequential(Transpose(1,2),norm_layer(dim), Transpose(1,2))        
        self.attn = Attention(dim, num_heads=num_heads,qkv_bias=True, qk_scale=False, attn_drop = drop, proj_drop=drop)
        self.drop_path = nn.Identity() #DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.norm2 = nn.Sequential(Transpose(1,2),norm_layer(dim),Transpose(1,2))   
        self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer = act_layer, drop = drop)

    def forward(self, x):
        z = self.norm1(x)
        z = self.attn(z)
        z = self.norm2(z)
        z = self.mlp(z)
        x = x+z
        return x 


class FlattenHead(nn.Module):
    def __init__(self, individual, c_in, nf, prediction_horizon, head_dropout = 0.):
        super(FlattenHead,self).__init__()
        self.individual = individual
        self.c_in = c_in

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(c_in):
                self.flattens.append(nn.Flatten(start_dim = -2))
                self.linears.append(nn.Linear(nf,prediction_horizon))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim = -2)
            self.linear = nn.Linear(nf, prediction_horizon)
            self.dropout = nn.Dropout(head_dropout)
    def forward(self,x):                                            # x: [bs x c_in x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.c_in):
                z = self.flattens[i](x[:,i,:,:])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim = -1)                        # x: [bs x c_in x prediction_horizon] 
        else:
            x = self.flatten(x)
            x =self.linear(x)
            x= self.dropout(x)
        return x
class PatchEmbed(nn.Module):
    def __init__(self, seq_length = 336, patch_size = 16, in_channels=1, embed_dim=512, norm_layer=None):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = int((seq_length - patch_size) / patch_size + 1)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # Trying out  linear projection instead of conv
        self.proj = nn.Linear(patch_size, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x): # x is of shape (B, C, L)
        x = x.unfold(-1, self.patch_size, self.patch_size)  # (B, C, num_patches, patch_size)  
        x = self.proj(x) # (B, C, embed_dim, num_patches)
        if self.norm:
            x = self.norm(x)
        return x


def positional_encoding(pe,learn_pe,q_len, d_model):
        # Positional encoding from patchTST github https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_layers.py
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class Model(nn.Module):
    def __init__(self,configs):
        super(Model,self).__init__()
        self.c_in = configs.enc_in
        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        drop_rate = configs.dropout
        revin = configs.revin 
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Patching
        self.patch_len = configs.patch_len
        patch_num  = int((self.seq_len - self.patch_len) / self.patch_len + 1)
        q_len = patch_num


        # head
        self.individual = configs.individual
        self.head_nf = d_model *patch_num

        if revin:
            self.revin = RevIN(num_features = self.c_in, eps= 1e-5, affine=configs.affine, subtract_last=configs.subtract_last)
        else:
            self.revin = None
        self.pos_embed = positional_encoding(configs.pos_embed, configs.learn_pos_embed,q_len , configs.d_model)
        self.pos_drop = nn.Dropout(p = drop_rate)
        self.patch_embed = PatchEmbed(seq_length = self.seq_len, patch_size = self.patch_len, in_channels=self.c_in, embed_dim=d_model)
        self.head = FlattenHead(self.individual,self.c_in,self.head_nf,self.pred_len,configs.head_dropout)

        alpha = configs.alpha
        self.blocks = nn.ModuleList()
        for i in range(configs.depth):
            if i < alpha:
                self.blocks.append(Block(dim=d_model, mlp_ratio=4., drop_path = drop_rate, act_layer=nn.GELU, norm_layer=nn.BatchNorm1d))
            else:
                self.blocks.append(Block_attention(dim=d_model, mlp_ratio=4., drop = drop_rate,drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm1d))
    def forward(self,x): # x is of shape (B,L,C)
        if self.revin:
            x = self.revin(x, 'norm')
        x = x.permute(0,2,1) # (B,C,L)
        x = self.patch_embed(x) # (B,C,d_model, num_patches)
        x = x + self.pos_embed 
        x = self.pos_drop(x)
        
        n_vars = x.shape[1]

        x = x.reshape(x.shape[0]*n_vars,x.shape[2],x.shape[3]) #(B*C, d_model, num_patches)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(-1,n_vars,x.shape[-2],x.shape[-1]) # (B,C,num_patches,d_model)
        x = x.permute(0,1,3,2)  #(B,C,d_model,num_patches)
        x = self.head(x) #(B,C,pred_len)
        if self.revin:
            x = x.permute(0,2,1)
            x = self.revin(x, 'denorm')
        return x
#model = Model(configs)
#def count_params(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#count_params(model)
