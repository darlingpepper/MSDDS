import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
import os
import torchvision
import torch
import torch.nn as nn


class AttentionFusion_auto(torch.nn.Module):
    def __init__(self, n_dim_input1, n_dim_input2, lambda_1=1, lambda_2=1):
        super(AttentionFusion_auto, self).__init__()
        self.n_dim_input1, self.n_dim_input2 = n_dim_input1, n_dim_input2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.linear = nn.Linear(2 * n_dim_input1, n_dim_input2)
    def forward(self, input1, input2): 
        mid_emb = torch.cat((input1, input2), 1)
        return F.relu(self.linear(mid_emb))

class CellCNN(nn.Module):
    def __init__(self, in_channel=6, feat_dim=128):
        super(CellCNN, self).__init__()
        max_pool_size=[2,2,6]
        drop_rate=0.2
        kernel_size=[16,16,16]
        if in_channel == 3:
            in_channels=[3,8,16]
            out_channels=[8,16,32]         
        elif in_channel == 6:
            in_channels=[6,16,32]
            out_channels=[16,32,64]
        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )
        self.cell_linear = nn.Linear(out_channels[2], feat_dim)

    def forward(self, x):
        # [b, genome_num(4076), 6]
        # print(x.shape)
        x = x.squeeze(1)
        b, g, n = x.shape
        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)  
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed) 
        # out: [b, 165, 128]
        x_cell_embed = x_cell_embed.view(b ,-1)
        # out: [b, 21120]
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)

class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """
    def __init__(self, embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num
        # 直接定义参数, 更加直观
        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))
        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))
        # 初始化, 避免计算得到nan
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)
    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """
        # 线性变换到注意力空间中
        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))
        # Head (head_num, bs, fields, D / head_num)
        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))
        # 计算内积
        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5
        # Softmax归一化权重
        attn_w = F.softmax(inner, dim=-1)
        #         attn_w = entmax15(inner, dim=-1)
        attn_w = F.dropout(attn_w, p=self.dropout)
        # 加权求和
        results = torch.matmul(attn_w, Value)
        # 拼接多头空间
        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)
        # 残差学习
        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))
        results = F.relu(results)
        return results

class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """
    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
            x = self.dropout(x)
        return x


class Interact(nn.Module):
    def __init__(self, field_dim, embed_size, dropout=0.5):
        super(Interact, self).__init__()
        self.bit_wise_net = Highway(input_size=field_dim * embed_size,
                                    num_highway_layers=2)
        hidden_dim = 1024
        self.trans_bit_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))
        m_bit = self.trans_bit_nn(bit_wise_x)
        m_x = m_bit + x.reshape(b, f * e)
        return m_x
    
    

class BothNet(nn.Module):
    def __init__(self, proj_dim=512, head_num=8, dropout_rate=0.5):
        super(BothNet, self).__init__()
        proj_dim = 512
        dropout_rate = 0.5
        self.feature_interact = Interact(field_dim=5, embed_size=proj_dim)
        self.cell_conv = CellCNN(in_channel=6, feat_dim=128)
        self.projection_cell = nn.Sequential(
            nn.Linear(21120, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
            )
        self.drug_feature_interact = MultiHeadAttentionInteract(embed_size=384,#128
                                                       head_num=head_num,#8
                                                       dropout=dropout_rate)
        self.projection_drug1_video = nn.Sequential(
            nn.Linear(6 * 384, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_drug2_video = nn.Sequential(
            nn.Linear(6 * 384, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_drug1_image = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_drug2_image = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim*5),
            nn.Linear(proj_dim*5, 2),
        )

    def forward(self, cell, drug1_2d_features, drug1_3d_features, drug2_2d_features, drug2_3d_features):
        cell = self.cell_conv(cell)
        cell = self.projection_cell(cell)
        drug1_2d_features = drug1_2d_features
        drug1_2d_features = self.projection_drug1_image(drug1_2d_features)
        drug2_2d_features = drug2_2d_features
        drug2_2d_features = self.projection_drug2_image(drug2_2d_features)
        b, f, e = drug1_3d_features.shape
        drug1_3d_features = self.drug_feature_interact(drug1_3d_features).reshape(b, (f*e))
        drug1_3d_features = self.projection_drug1_video(drug1_3d_features)
        drug2_3d_features = self.drug_feature_interact(drug2_3d_features).reshape(b, (f*e))
        drug2_3d_features = self.projection_drug2_video(drug2_3d_features)
        all_features = torch.stack([drug1_2d_features, drug1_3d_features, cell, drug2_2d_features, drug2_3d_features], dim=1)
        all_features = self.feature_interact(all_features)
        out = self.transform(all_features)
        return out
