import torch
import torch.nn as nn
import e3nn
from e3nn import o3
from e3nn.nn import Gate
from e3nn.math import soft_one_hot_linspace

from torch_scatter import scatter
from torch_cluster import radius


class FinetuneBlock(nn.Module):
    def __init__(self, configs):
        """参数说明
        Args:
            configs: 全局配置对象, 需要包含
                configs.c_atom: 主干 single 表征通道数 (默认 128)
                configs.glue_env_encoder.*: LocalEnvEncoder 超参
        """
        super().__init__()

        # 保存常用维度
        # self.c_s_inputs: int = configs.c_s_inputs  # single 维度, 与 PairFormer 输出一致
        self.c_atom = configs.c_atom

        # --- 子模块 ---
        self.glue_env_encoder = LocalEnvEncoder(configs)

        # Cross-attention 的维度来自配置或默认值
        self.cross_attn = E3NNCrossAttn(
            configs,
            ligand_scalar_dim=self.glue_env_encoder.out_scalar_dim,
            ligand_vector_dim=self.glue_env_encoder.edge_out_vector_dim,
            num_layers=getattr(configs, "finetune_cross_attn_layers", 4),
            num_heads=getattr(configs, "finetune_cross_attn_heads", 4),
        )

        # 线性 Adapter, 初始为零, 仅微调输出幅度
        self.adapter = nn.Linear(self.c_atom, self.c_atom)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, input_feature_dict, x_gt_augment, atom_level_s, current_x, current_t):
        """
        Forward pass of the FinetuneBlock.
        Args:
            input_feature_dict: dict, input feature dict
            x_gt_augment: torch.Tensor, augmented x_gt, true value
            atom_level_s: torch.Tensor, atom level s
            current_x: torch.Tensor, x before denoise
            current_sigma: torch.Tensor, noise level
        """
        atom_level_s = atom_level_s.detach()
        k_value = 1   # schedule according to training step or loss
        # teacher forcing
        x_gt_augment = k_value * x_gt_augment + (1 - k_value) * current_x
        x_gt_augment = x_gt_augment.detach()

        # 1. Encode glue environment (batched over N_sample)
        B = x_gt_augment.shape[-3]           # N_sample

        glue_scalar_list = []
        glue_vector_list = []
        for i in range(B):
            gs, gv = self.glue_env_encoder(
                pos=x_gt_augment[i],                       # (N_atom,3)
                elem_id=input_feature_dict["ref_element"],
                charge=input_feature_dict["ref_charge"],
                center_mask=input_feature_dict["is_glue"],
            )
            glue_scalar_list.append(gs)     # (N_glue,Cs)
            glue_vector_list.append(gv)     # (N_glue,Cv,3)

        glue_env_scalar_b = torch.stack(glue_scalar_list, dim=0)   # (B,N_glue,Cs)
        glue_env_vector_b = torch.stack(glue_vector_list, dim=0)   # (B,N_glue,Cv,3)

        # 2. Broadcast pairformer_s to batch dimension
        atom_level_s_b = atom_level_s.expand(B, -1, -1)   # (B,N_token,c_s)

        # 3. Cross-attention
        updated_scalar_b, delta_pos_b = self.cross_attn(
            global_scalar=atom_level_s_b,
            ligand_scalar=glue_env_scalar_b,
            ligand_vector=glue_env_vector_b,
        )

        # 保持批维度 (B, N_token, ...)
        updated_scalar = updated_scalar_b            # (B,N_token,c_atom)
        delta_pos = delta_pos_b                      # (B,N_token,3)

        # 3. Adapter
        updated_scalar = self.adapter(updated_scalar)

        return updated_scalar, delta_pos

class LocalEnvEncoder(nn.Module):
    """Encode glue 环境 → 每个 glue 原子输出 0e+1o 表征。

    输出:
        scalar : (n_glue, out_scalar_dim)
        vector : (n_glue, out_vector_dim, 3)
    """

    def __init__(
        self,
        configs
    ):
        super().__init__()

        # ------------ hyper parameters -------------
        self.cutoff = configs.glue_env_encoder.cutoff    # 5 Å
        self.n_rbf  = configs.glue_env_encoder.n_rbf
        self.lmax   = configs.glue_env_encoder.lmax      # 1

        self.edge_out_scalar_dim = configs.glue_env_encoder.ligand_encoder_out_scalar_dim
        self.center_out_scalar_dim = configs.glue_env_encoder.scalar_encoder_out_scalar_dim
        self.out_scalar_dim = self.edge_out_scalar_dim + self.center_out_scalar_dim

        self.edge_out_vector_dim = configs.glue_env_encoder.ligand_encoder_out_vector_dim

        # ------------ node embedding (0e) ----------
        self.elem_emb = nn.Embedding(118, 10)
        self.feat_lin = nn.Linear(11, configs.glue_env_encoder.scalar_encoder_out_scalar_dim, bias=False)    # 10+1 → scalar_encoder_out_scalar_dim×0e
        self.node_irreps = o3.Irreps(f"{configs.glue_env_encoder.scalar_encoder_out_scalar_dim}x0e")

        # ------------ edge irreps ------------------
        edge_irreps = o3.Irreps(f"{self.n_rbf}x0e + {self.lmax}x1o")

        # ---------- message & gate irreps ----------
        # 结构: (out_scalar_dim)x0e  + (out_vector_dim)x0e  + (out_vector_dim)x1o
        self.msg_irreps = o3.Irreps(
            f"{self.edge_out_scalar_dim}x0e + {self.edge_out_vector_dim}x0e + {self.edge_out_vector_dim}x1o"
        )


        # Fully-connected tensor product: node ⊗ edge → message
        self.tp_msg = o3.FullyConnectedTensorProduct(
            self.node_irreps,   # scalar_encoder_out_scalar_dim×0e
            edge_irreps,        # n_rbf x0e + 1x1o
            self.msg_irreps
        )

        # Gate 构造: scalars → ReLU, gates → sigmoid, vectors → tanh*gate
        self.gate = Gate(
            f"{self.edge_out_scalar_dim}x0e",          # scalars
            [torch.nn.SiLU()],
            f"{self.edge_out_vector_dim}x0e",          # gates
            [torch.sigmoid],
            f"{self.edge_out_vector_dim}x1o"           # gated vectors (odd parity)
        )

    def forward(
        self,
        pos: torch.Tensor,               # (N,3)
        elem_id: torch.Tensor,           # (N,)
        charge: torch.Tensor,            # (N,)
        center_mask: torch.Tensor,       # (N,) bool
    ):
        N = pos.size(0)
        center_idx = center_mask.nonzero(as_tuple=True)[0]  # (n_glue,)

        # -------- build env→center edges --------
        # torch_cluster.radius: given point sets x (N_env) and y (N_center),
        # returns (row, col) s.t. ||x[row] - y[col]|| < r
        # We want edges from env(atom) -> center(glue). So set x=pos, y=pos[center_idx].

        edge_center, edge_env = radius(
            x=pos, y=pos[center_idx], r=self.cutoff, max_num_neighbors=64
        )  # (E,), each indexing into respective sets


        # Convert local center indices (0..n_glue-1) to global atom indices
        global_center_idx = center_idx[edge_center]          # (E,)

        rel = pos[global_center_idx] - pos[edge_env]         # (E,3)  direction: env -> center
        d = rel.norm(dim=-1) + 1e-8                          # avoid zero div

        # ---- edge features 0e+1o ----
        rbf = soft_one_hot_linspace(
            d, 0.0, self.cutoff, self.n_rbf, basis="cosine", cutoff=True
        )  # (E,n_rbf)

        Ylm = o3.spherical_harmonics(
            [1], rel / d.unsqueeze(-1), normalize=True, normalization="component"
        )  # (E,3)

        edge_feat = torch.cat([rbf, Ylm], dim=-1)

        # ---- node scalar embedding ----
        charge_expanded = charge.unsqueeze(-1)
        elem_id = elem_id.argmax(dim=-1)
        node_scalar = torch.cat([self.elem_emb(elem_id), charge_expanded], dim=-1)
        node_feat = self.feat_lin(node_scalar)  # (N,scalar_encoder_out_scalar_dim)

        # ---- message passing ----
        m = self.tp_msg(node_feat[edge_env], edge_feat)  # (E, 48)
        m = self.gate(m)                                # (E, out_irreps)

        # aggregate to center (local index edge_center)
        agg = scatter(m, edge_center, dim=0, dim_size=center_idx.size(0), reduce="mean")

        # 返回两个张量方便后续 cross-attn 使用
        scalar_dim = self.edge_out_scalar_dim
        vector_dim = self.edge_out_vector_dim
        
        scalar = agg[:, :scalar_dim]                     # (n_glue, ligand_out_scalar_dim)
        center_feat = node_feat[center_idx]
        scalar = torch.cat([center_feat, scalar], dim=-1)

        vector = agg[:, scalar_dim:].reshape(-1, vector_dim, 3)                     # (n_glue, out_vector_dim, 3)

        return scalar, vector

class E3NNCrossAttn(nn.Module):
    """Cross-attention block(s) between protein scalar embeddings (0e) and ligand/glue
    scalar + vector (0e + 1o) representations.

    This implementation stacks *num_layers* identical blocks.  每层流程：
        1. 仅使用标量部分 (0e) 计算 Query / Key，得到注意力权重。
        2. 用权重对 (0e + 1o) Value 做加权，得到融合后的 (0e + 1o)。
        3. 通过线性头把输出 1o 投影成 **单个方向向量 (3) ≡ 1x1o**，作为坐标残差。

    参数建议（若调用者未指定）：
        global_scalar_dim  : 128   # 对应主干 c_atom 的标量通道数
        ligand_scalar_dim  :  32   # LocalEnvEncoder 输出 0e 通道数
        ligand_vector_dim  :   8   # LocalEnvEncoder 输出 1o 通道数（每个 1o = 3 实数 → 24 维）
        num_layers         :   4   # 堆叠层数
        num_heads          :   4   # 与 Protenix trunk 保持一致
    """
    def __init__(
        self,
        configs,
        ligand_scalar_dim: int = 32,
        ligand_vector_dim: int = 8,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.global_scalar_dim = configs.c_atom
        self.ligand_scalar_dim = ligand_scalar_dim
        self.ligand_vector_dim = ligand_vector_dim  # number of 1o channels (each 3-D)

        # self.query_proj = nn.Linear(self.global_scalar_dim, configs.c_s)
        # ---------------- Scalar Attention ----------------
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=configs.c_atom,
                    kdim=ligand_scalar_dim,
                    vdim=ligand_scalar_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )


        # --- vector projection: equivariant 1o→1o (IrrepsArray not used) ---
        in_irreps = o3.Irreps(f"{ligand_vector_dim}x1o")
        out_irreps = o3.Irreps("1x1o")
        self.proj_vector = nn.ModuleList(
            [o3.Linear(in_irreps, out_irreps) for _ in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(configs.c_atom) for _ in range(num_layers)]
        )

    def forward(
        self,
        global_scalar: torch.Tensor,  # (B, N_prot, global_scalar_dim)
        ligand_scalar: torch.Tensor,  # (B, N_glue, ligand_scalar_dim)
        ligand_vector: torch.Tensor,  # (B, N_glue, ligand_vector_dim, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run stacked cross-attention.

        Returns
        --------
        updated_scalar : (B, N_prot, global_scalar_dim)
        delta_pos      : (B, N_prot, 3)  cumulative direction residual
        """
        B, N_prot, _ = global_scalar.shape
        delta_pos_total = torch.zeros(B, N_prot, 3, device=global_scalar.device, dtype=global_scalar.dtype)
        # h = self.query_proj(global_scalar)
        h = global_scalar

        # ligand_vector 保持形状 (B, N_glue, ligand_vector_dim, 3)，方便做 IrrepsArray

        for i in range(self.num_layers):
            # ---- Multi-head Attention on scalar part ----
            attn_out, attn_weights = self.attn_layers[i](
                query=h, key=ligand_scalar, value=ligand_scalar, need_weights=True
            )
            h = h + attn_out  # residual proj
            h = self.layer_norms[i](h)

            # ---- Direction residual ----
            # 重新计算注意力权重（需要权重，这里用 dot-product same as above）
            # attn_scores = torch.einsum(
            #     "bij,bkj->bik",  # (B,N,D)·(B,M,D)
            #     h,
            #     ligand_scalar,
            # ) / (self.global_scalar_dim ** 0.5)
            # attn_weights = torch.softmax(attn_scores, dim=-1)  # (B,N_prot,N_glue)

            # 加权求和 vector → (B,N_prot,ligand_vector_dim,3)
            weighted_vec = torch.einsum("bij,bjkc->bikc", attn_weights, ligand_vector)  # (B,N,K,3)

            # 直接使用张量接口（老版 e3nn 无 IrrepsArray）
            weighted_vec_flat = weighted_vec.reshape(
                weighted_vec.shape[0], weighted_vec.shape[1], -1
            )  # (B,N,K*3)
            delta = self.proj_vector[i](weighted_vec_flat)  # (B,N,3)
            delta_pos_total = delta_pos_total + delta

        return h, delta_pos_total 