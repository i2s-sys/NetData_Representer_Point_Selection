import torch
import torch.nn as nn


class EmbeddingModule(nn.Module):  # 嵌入层
    def __init__(self, num_user, num_item, num_time, latent_dim):  
        super().__init__()
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim))) 

    def forward(self, i_input, j_input, ks_input):
        i_embed = self.user_embeddings[i_input]
        j_embed = self.item_embeddings[j_input]
        k_embed_seq = self.time_embeddings[ks_input]  
        return i_embed, j_embed, k_embed_seq


class PatchTransformer(nn.Module):  # 时间序列 patch 提取与 Transformer 编码
    def __init__(self, latent_dim, patch_len, stride, max_num_patches, num_layers, nhead):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(latent_dim, latent_dim)

        # patch间的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # patch内部的位置编码（作为fallback）
        self.inner_pos_embed = nn.Parameter(torch.zeros(1, patch_len, latent_dim))
        nn.init.trunc_normal_(self.inner_pos_embed, std=0.02)
        
        # 绝对时间位置编码 - 使用正弦位置编码的思想
        self.time_scale = nn.Parameter(torch.ones(1))
        self.time_pos_proj = nn.Linear(latent_dim, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead,
            dim_feedforward=latent_dim * 2, dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # patch内部的注意力聚合
        self.inner_attn = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softmax(dim=2)
        )

        self.pooling_attn = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softmax(dim=1)
        )
        self.layer_norm1 = nn.LayerNorm(latent_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)

    def get_time_position_encoding(self, time_indices):
        """
        基于绝对时间戳生成位置编码
        time_indices: [B, P, patch_len] 绝对时间戳
        """
        B, P, patch_len = time_indices.shape
        D = self.inner_pos_embed.size(-1)
        
        # 使用正弦位置编码公式，但基于真实时间戳
        position = time_indices.unsqueeze(-1).float()  # [B, P, patch_len, 1]
        
        # 创建维度索引 - 确保维度匹配
        div_term = torch.exp(torch.arange(0, D, 2, device=time_indices.device, dtype=torch.float) * 
                           -(torch.log(torch.tensor(10000.0)) / D))
        div_term = div_term * self.time_scale  # 可学习的时间缩放因子
        
        # 生成位置编码
        pos_encoding = torch.zeros(B, P, patch_len, D, device=time_indices.device)
        
        # 处理偶数维度（sin）
        even_dims = torch.arange(0, D, 2, device=time_indices.device)
        if len(even_dims) > 0:
            pos_encoding[:, :, :, even_dims] = torch.sin(position * div_term[:len(even_dims)])
        
        # 处理奇数维度（cos）
        odd_dims = torch.arange(1, D, 2, device=time_indices.device)
        if len(odd_dims) > 0:
            pos_encoding[:, :, :, odd_dims] = torch.cos(position * div_term[:len(odd_dims)])
        
        return self.time_pos_proj(pos_encoding)

    def make_patches(self, x, time_indices=None):
        B, L, D = x.shape
        patches = [x[:, i:i + self.patch_len, :] for i in range(0, L - self.patch_len + 1, self.stride)]
        
        if time_indices is not None:
            # 提取对应的时间戳patches
            time_patches = [time_indices[:, i:i + self.patch_len] for i in range(0, L - self.patch_len + 1, self.stride)]
            return torch.stack(patches, dim=1), torch.stack(time_patches, dim=1)  # [B, P, patch_len, D], [B, P, patch_len]
        
        return torch.stack(patches, dim=1)  # [B, P, patch_len, D]

    def forward(self, k_embed_seq, time_indices=None):
        if time_indices is not None:
            patches, time_patches = self.make_patches(k_embed_seq, time_indices)
            B, P, patch_len, D = patches.shape
            
            # 使用绝对时间戳计算位置编码
            time_pos_encoding = self.get_time_position_encoding(time_patches)
            patches = patches + time_pos_encoding
        else:
            patches = self.make_patches(k_embed_seq)
            B, P, patch_len, D = patches.shape
            
            # 使用默认的位置编码
            patches = patches + self.inner_pos_embed[:, :patch_len, :]
        
        # 使用注意力机制聚合patch内部信息
        patches_reshaped = patches.view(B * P, patch_len, D)
        inner_weights = self.inner_attn(patches_reshaped)
        patches_pooled = (patches_reshaped * inner_weights).sum(dim=1)
        patches = patches_pooled.view(B, P, D)
        
        patches = self.patch_proj(patches)
        patches = self.layer_norm1(patches)
        
        # 添加patch间位置编码
        patches = patches + self.pos_embed[:, :patches.shape[1], :]
        patches = self.transformer(patches)
        
        patches = self.layer_norm2(patches)
        
        weights = self.pooling_attn(patches)
        pooled = (patches * weights).sum(dim=1)
        return pooled

class InteractionModule(nn.Module):#两两交互建模
    def __init__(self, latent_dim):
        super().__init__()
        reduced_dim = latent_dim // 2
        self.i_proj = nn.Linear(latent_dim, reduced_dim)
        self.j_proj = nn.Linear(latent_dim, reduced_dim)
        self.k_proj = nn.Linear(latent_dim, reduced_dim)
        self.reduced_dim = reduced_dim

    def pairwise_outer(self, a, b):
        outer = torch.einsum('bd,be->bde', a, b)
        return outer.view(a.size(0), -1)

    def forward(self, i_embed, j_embed, k_embed):
        i = self.i_proj(i_embed)
        j = self.j_proj(j_embed)
        k = self.k_proj(k_embed)
        ij = self.pairwise_outer(i, j)
        ik = self.pairwise_outer(i, k)
        jk = self.pairwise_outer(j, k)
        return ij, ik, jk


class FinalMLP(nn.Module):#最终 MLP 层
    def __init__(self, latent_dim):
        super().__init__()
        reduced_dim = latent_dim // 2
        in_dim = 3 * reduced_dim * reduced_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, features):
        return self.mlp(features).squeeze(-1)

class CP(nn.Module):
    def __init__(self, num_dim, latent_dim):
        super(CP, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

    def forward(self, i_input, j_input, k_input, ks_input=None):
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[k_input]
        xijk = torch.einsum('nd, nd, nd -> n', i_embeds, j_embeds, k_embeds)
        return xijk
    
class CP_A(nn.Module):
    def __init__(self, num_dim, latent_dim, patch_len=8, stride=4, num_layers=2, nhead=4, max_num_patches=64):
        super(CP_A, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

        self.patch_transformer = PatchTransformer(
            latent_dim, patch_len, stride, max_num_patches, num_layers, nhead
        )

    def forward(self, i_input, j_input, k_input, ks_input=None):
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_seq = self.time_embeddings[ks_input]

        if ks_input is None:
            time_indices = ks_input.float()
        else:
            time_indices = None

        k_embeds = self.patch_transformer(k_seq, time_indices)
        xijk = torch.einsum('nd, nd, nd -> n', i_embeds, j_embeds, k_embeds)
        return xijk
    
class CP_B(nn.Module):
    def __init__(self, num_dim, latent_dim):
        super(CP_B, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

        self.interaction = InteractionModule(latent_dim)
        self.final_mlp = FinalMLP(latent_dim)

    def forward(self, i_input, j_input, k_input, ks_input=None):
        i_emb = self.user_embeddings[i_input]
        j_emb = self.item_embeddings[j_input]
        k_emb = self.time_embeddings[k_input]

        ij, ik, jk = self.interaction(i_emb, j_emb, k_emb)
        features = torch.cat([ij, ik, jk], dim=-1)
        return self.final_mlp(features)

