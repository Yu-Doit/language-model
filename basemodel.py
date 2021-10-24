import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0., bias = True, add_bias_kv = False,
                 add_zero_attn = False, kdim = None, vdim = None, device = None, dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def _in_projection_packed(self, q, k, v):
        E = q.size(-1)
        if k is v:
            if q is k:
                return F.linear(q, self.in_proj_weight, self.in_proj_bias).chunk(3, dim = -1)
            else:
                w_q, w_kv = self.in_proj_weight.split([E, E * 2])
                if self.in_proj_bias is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = self.in_proj_bias.split([E, E * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim = -1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def _in_projection(self, q, k, v, b_q, b_k, b_v):
        return F.linear(q, self.q_proj_weight, b_q), F.linear(k, self.k_proj_weight, b_k), F.linear(v, self.v_proj_weight, b_v)


    def _forward(self, query, key, value, training = True, key_padding_mask = None, need_weights = True,
                 attn_mask = None, use_separate_proj_weight = False, static_k = None, static_v = None):
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        if not use_separate_proj_weight:
            q, k, v = self._in_projection_packed(query, key, value)
        else:
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            q, k, v = self._in_projection(query, key, value, b_q, b_k, b_v)

        if self.bias_k is not None and self.bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            v = static_v

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype = k.dtype, device = k.device)], dim = 1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype = k.dtype, device = k.device)], dim = 1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        
        src_len = k.size(1)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))

        dropout_p = self.dropout
        if not self.training:
            dropout_p = 0.0

        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim = 1) / self.num_heads
        else:
            return attn_output, None

    def forward(self, query, key, value, key_padding_mask = None, need_weights = True, attn_mask = None):
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self._forward(
                query, key, value, key_padding_mask = key_padding_mask,
                need_weights = need_weights, attn_mask = attn_mask, use_separate_proj_weight = True)
        else:
            attn_output, attn_output_weights = self._forward(
                query, key, value, key_padding_mask = key_padding_mask,
                need_weights = need_weights, attn_mask = attn_mask)
        
        return attn_output, attn_output_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1, activation = F.relu,
                 layer_norm_eps = 1e-5, norm_first = False, device = None, dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout = dropout, **factory_kwargs)

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps, **factory_kwargs)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        
        super(TransformerEncoderLayer, self).__setstate__(state)

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask = attn_mask,
            key_padding_mask = key_padding_mask, need_weights = False)[0]
        return self.dropout(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask, src_key_padding_mask):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm = None):
        super(TransformerEncoder, self).__init__()
        
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)

        return output


def _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p = 0.0):
    b, Nt, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim = -1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p = dropout_p)
    output = torch.bmm(attn, v)
    return output, attn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu