import argparse
import copy
import csv
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


# QM9 标签索引：homo=2, lumo=3（PyG QM9 标准顺序）
QM9_HOMO_LUMO_INDICES = [2, 3]
QM9_TARGET_NAMES = ("homo", "lumo")

# Matplotlib 中文和负号兼容配置，适配 Windows 中文环境。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 工具函数1：SMILES -> RDKit 分子 2D 图像 -> Tensor
# ============================================================

def smiles_to_rdkit_image_tensor(
    smiles: str,
    image_size: int = 128,
    add_hs: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """
    将 SMILES 字符串转为 RDKit 2D 分子结构图像，再转为 PyTorch Tensor。

    返回：
        img_tensor: shape = [3, image_size, image_size]
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    '''mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")'''
    # 关闭严格sanitize，兼容QM9特殊环编号SMILES
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # 解析失败就返回空白全黑图像张量，不抛错中断训练
    if mol is None:
        return torch.zeros(3, 128, 128, dtype=torch.float32)

    # 选择性做基础清洗，避免部分价态报错
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FIND_RINGS)
    except:
        pass

    # 化学家先验知识输入：可选显式氢，然后生成 2D 坐标。
    if add_hs:
        mol = Chem.AddHs(mol)

    AllChem.Compute2DCoords(mol)

    # RDKit 生成 PIL RGB 图像；Pillow 是 RDKit/torchvision 常规运行依赖。
    pil_img = Draw.MolToImage(mol, size=(image_size, image_size)).convert("RGB")

    transform_steps = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # [H, W, C] / PIL -> float tensor [C, H, W], range [0, 1]
    ]

    if normalize:
        # 使用 ImageNet 风格归一化，使 CNN 输入数值稳定。
        transform_steps.append(
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        )

    transform = T.Compose(transform_steps)
    return transform(pil_img).float()


# ============================================================
# 工具函数2：QM9 节点特征 + 位置编码 Pm 构建
# ============================================================

def _one_hot_atomic_number(z: torch.Tensor, max_atomic_num: int = 100) -> torch.Tensor:
    """
    当 Data.x 不存在时，用原子序数 z 构造 one-hot 节点特征。
    """
    z = z.long().clamp(min=0, max=max_atomic_num)
    return F.one_hot(z, num_classes=max_atomic_num + 1).float()


def _centered_position_encoding(
    pos: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    构造位置编码 Pm：
    - 每个分子内将 3D 坐标中心化
    - 拼接 radial distance = ||centered_xyz||
    - 输出 [num_nodes, 4]
    """
    pos = pos.float()

    if batch is None:
        centroid = pos.mean(dim=0, keepdim=True)
        centered = pos - centroid
    else:
        batch = batch.long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        counts = torch.bincount(batch, minlength=num_graphs).float().clamp_min(1.0)
        centroid_sum = torch.zeros(
            num_graphs, pos.size(-1), device=pos.device, dtype=pos.dtype
        )
        centroid_sum.index_add_(0, batch, pos)
        centroids = centroid_sum / counts.unsqueeze(-1)
        centered = pos - centroids[batch]

    radial = torch.norm(centered, p=2, dim=-1, keepdim=True)
    return torch.cat([centered, radial], dim=-1)


def build_node_features_with_position_encoding(
    data: Data,
    max_atomic_num: int = 100,
) -> torch.Tensor:
    """
    QM9 节点特征 + 位置编码构建函数。

    输入：
        data.x   : PyG QM9 原始节点特征，若不存在则由 data.z 生成 one-hot
        data.pos : QM9 3D 原子坐标

    输出：
        node_feat_with_pos: [num_nodes, base_node_dim + 4]
    """
    if hasattr(data, "x") and data.x is not None:
        base_x = data.x.float()
    elif hasattr(data, "z") and data.z is not None:
        base_x = _one_hot_atomic_number(data.z, max_atomic_num=max_atomic_num)
    else:
        raise ValueError("Data object must contain either `x` or `z` for node features.")

    if hasattr(data, "pos") and data.pos is not None:
        batch = data.batch if hasattr(data, "batch") else None
        pos_enc = _centered_position_encoding(data.pos, batch=batch).to(base_x.device)
    else:
        pos_enc = torch.zeros(
            base_x.size(0), 4, dtype=base_x.dtype, device=base_x.device
        )

    return torch.cat([base_x, pos_enc], dim=-1)


# ============================================================
# 工具函数3：单 SMILES 推理数据构建
# ============================================================


def _parse_smiles_for_inference(smiles: str) -> "Chem.Mol":
    """
    仅用于推理的严格 SMILES 解析。
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}") from exc

    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")

    return mol


def _rdkit_atom_feature_vector(atom: "Chem.Atom") -> List[float]:
    """
    构建与 QM9 单样本推理兼容的 11 维原子基础特征。
    """
    symbol = atom.GetSymbol()
    hybridization = atom.GetHybridization()

    return [
        float(symbol == "H"),
        float(symbol == "C"),
        float(symbol == "N"),
        float(symbol == "O"),
        float(symbol == "F"),
        float(atom.GetAtomicNum()),
        float(atom.GetIsAromatic()),
        float(hybridization == Chem.HybridizationType.SP),
        float(hybridization == Chem.HybridizationType.SP2),
        float(hybridization == Chem.HybridizationType.SP3),
        float(atom.GetTotalNumHs(includeNeighbors=True)),
    ]


def _build_inference_node_features(
    mol: Chem.Mol,
    base_dim: int,
) -> torch.Tensor:
    """
    根据 checkpoint 的 node_input_dim 生成与训练阶段兼容的基础节点特征。
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return torch.zeros((0, max(base_dim, 0)), dtype=torch.float32)

    if base_dim == 101:
        return torch.stack(
            [
                _one_hot_atomic_number(torch.tensor(atom.GetAtomicNum()), max_atomic_num=100)
                for atom in mol.GetAtoms()
            ],
            dim=0,
        ).float()

    base_features = torch.tensor(
        [_rdkit_atom_feature_vector(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )

    if base_dim <= 0:
        return torch.zeros((num_atoms, 0), dtype=torch.float32)

    if base_features.size(-1) < base_dim:
        padding = torch.zeros(
            num_atoms,
            base_dim - base_features.size(-1),
            dtype=base_features.dtype,
        )
        base_features = torch.cat([base_features, padding], dim=-1)
    elif base_features.size(-1) > base_dim:
        base_features = base_features[:, :base_dim]

    return base_features


def _rdkit_bond_feature_vector(bond: Chem.Bond) -> List[float]:
    bond_type = bond.GetBondType()
    features = [0.0, 0.0, 0.0, 0.0]

    if bond_type == Chem.BondType.SINGLE:
        features[0] = 1.0
    elif bond_type == Chem.BondType.DOUBLE:
        features[1] = 1.0
    elif bond_type == Chem.BondType.TRIPLE:
        features[2] = 1.0
    elif bond_type == Chem.BondType.AROMATIC:
        features[3] = 1.0
    else:
        features[0] = 1.0

    return features


def _fit_edge_attr_dim(edge_attr: torch.Tensor, edge_dim: Optional[int]) -> torch.Tensor:
    if edge_dim is None or edge_attr.size(-1) == edge_dim:
        return edge_attr

    if edge_attr.size(-1) < edge_dim:
        padding = torch.zeros(
            edge_attr.size(0),
            edge_dim - edge_attr.size(-1),
            dtype=edge_attr.dtype,
        )
        return torch.cat([edge_attr, padding], dim=-1)

    return edge_attr[:, :edge_dim]


def _build_inference_edges(
    mol: Chem.Mol,
    edge_dim: Optional[int] = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[int] = []
    cols: List[int] = []
    edge_features: List[List[float]] = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_feat = _rdkit_bond_feature_vector(bond)

        rows.extend([begin_idx, end_idx])
        cols.extend([end_idx, begin_idx])
        edge_features.extend([bond_feat, bond_feat])

    if not rows:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return edge_index, _fit_edge_attr_dim(edge_attr, edge_dim)


def _build_inference_positions(mol: Chem.Mol) -> torch.Tensor:
    """
    为推理构建 3D 坐标；失败时回退到全零坐标。
    """
    mol_with_hs = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    try:
        status = AllChem.EmbedMolecule(mol_with_hs, params)
    except Exception:
        status = -1

    if status != 0:
        return torch.zeros((mol_with_hs.GetNumAtoms(), 3), dtype=torch.float32)

    try:
        AllChem.UFFOptimizeMolecule(mol_with_hs)
    except Exception:
        pass

    conformer = mol_with_hs.GetConformer()
    positions = np.asarray(conformer.GetPositions(), dtype=np.float32)
    return torch.tensor(positions, dtype=torch.float32)


def smiles_to_inference_data(
    smiles: str,
    node_input_dim: int,
    image_size: int,
    edge_dim: Optional[int] = None,
    normalize_img: bool = True,
) -> Data:
    """
    将单个 SMILES 构建为可直接送入 KnowledgeAugGAT 的 PyG Data。
    """
    mol = _parse_smiles_for_inference(smiles)
    mol = Chem.AddHs(mol)

    base_dim = max(int(node_input_dim) - 4, 0)
    base_x = _build_inference_node_features(mol, base_dim=base_dim)
    pos = _build_inference_positions(mol)
    edge_index, edge_attr = _build_inference_edges(mol, edge_dim=edge_dim)

    data = Data(
        x=base_x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.x = build_node_features_with_position_encoding(data)
    data.img = smiles_to_rdkit_image_tensor(
        smiles,
        image_size=image_size,
        normalize=normalize_img,
    ).unsqueeze(0)
    data.smiles = smiles
    return data


# ============================================================
# 工具函数4：QM9 自动加载 + RDKit 图像懒加载包装
# ============================================================

def _safe_get_data_attr(data: Data, name: str):
    try:
        return getattr(data, name)
    except Exception:
        return None


class QM9WithRDKitImages(Dataset):
    """
    包装 PyG QM9：
    - 自动把每个样本的节点特征替换为：原始节点特征 + 位置编码 Pm
    - 自动从 SMILES 或 raw SDF 生成 RDKit 分子 2D 图像 Tensor
    - 为 PyG Batch 拼接方便，单样本 img 存为 [1, 3, H, W]
    """

    def __init__(
        self,
        base_dataset: QM9,
        image_size: int = 128,
        add_hs: bool = False,
        normalize_img: bool = True,
        max_atomic_num: int = 100,
    ):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.add_hs = add_hs
        self.normalize_img = normalize_img
        self.max_atomic_num = max_atomic_num

        self._image_cache: Dict[str, torch.Tensor] = {}
        self._sdf_smiles_cache: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _extract_smiles_from_data(self, data: Data, dataset_index: int) -> str:
        """
        优先使用 data.smiles；如果 PyG 版本未保存 smiles，则从 QM9 raw SDF 按 idx 取出。
        """
        for attr_name in ("smiles", "smile", "SMILES"):
            value = _safe_get_data_attr(data, attr_name)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, (list, tuple)) and len(value) > 0:
                first_value = value[0]
                if isinstance(first_value, str) and first_value:
                    return first_value

        if self._sdf_smiles_cache is None:
            self._sdf_smiles_cache = self._read_smiles_from_raw_sdf()

        # PyG QM9 通常保留原始 idx；若无 idx，就退回 dataset_index。
        raw_idx_value = _safe_get_data_attr(data, "idx")
        if raw_idx_value is not None:
            raw_idx = int(raw_idx_value.item()) if torch.is_tensor(raw_idx_value) else int(raw_idx_value)
        else:
            raw_idx = int(dataset_index)

        if raw_idx < 0 or raw_idx >= len(self._sdf_smiles_cache):
            raise IndexError(
                f"Cannot map QM9 data index {dataset_index} / raw idx {raw_idx} "
                f"to raw SDF smiles cache of length {len(self._sdf_smiles_cache)}."
            )

        smiles = self._sdf_smiles_cache[raw_idx]
        if not smiles:
            raise ValueError(f"Empty SMILES recovered from QM9 raw SDF at raw idx {raw_idx}.")
        return smiles

    def _read_smiles_from_raw_sdf(self) -> List[str]:
        """
        从 QM9 raw SDF 中恢复 SMILES。
        """
        raw_dir = getattr(self.base_dataset, "raw_dir", None)
        if raw_dir is None:
            raise RuntimeError("Cannot locate QM9 raw_dir to recover SMILES.")

        sdf_candidates = sorted(glob.glob(os.path.join(raw_dir, "*.sdf")))
        if not sdf_candidates:
            raise FileNotFoundError(
                f"No .sdf file found in QM9 raw_dir={raw_dir!r}. "
                "Please ensure QM9 was downloaded correctly."
            )

        # QM9 下载文件通常为 gdb9.sdf。
        sdf_path = None
        for path in sdf_candidates:
            if "gdb9" in os.path.basename(path).lower():
                sdf_path = path
                break
        if sdf_path is None:
            sdf_path = sdf_candidates[0]

        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        smiles_list: List[str] = []

        for mol in supplier:
            if mol is None:
                smiles_list.append("")
            else:
                try:
                    no_h_mol = Chem.RemoveHs(mol)
                    smiles_list.append(Chem.MolToSmiles(no_h_mol, canonical=True))
                except Exception:
                    smiles_list.append(Chem.MolToSmiles(mol, canonical=True))

        return smiles_list

    def __getitem__(self, index: int) -> Data:
        data = self.base_dataset[index]
        data = copy.copy(data)

        # 节点特征 + 位置编码 Pm。
        data.x = build_node_features_with_position_encoding(
            data,
            max_atomic_num=self.max_atomic_num,
        )

        # SMILES -> RDKit 2D 图像 Tensor。
        smiles = self._extract_smiles_from_data(data, dataset_index=index)
        if smiles not in self._image_cache:
            self._image_cache[smiles] = smiles_to_rdkit_image_tensor(
                smiles,
                image_size=self.image_size,
                add_hs=self.add_hs,
                normalize=self.normalize_img,
            )

        # PyG Batch 默认沿 dim=0 拼接，因此这里存成 [1, 3, H, W]。
        data.img = self._image_cache[smiles].unsqueeze(0)
        data.smiles = smiles
        data.y = data.y.view(-1)[QM9_HOMO_LUMO_INDICES]
        return data


def load_qm9_dataset(
    root: str = "./data/QM9",
    image_size: int = 128,
    transform=None,
    pre_transform=None,
    add_hs: bool = False,
    normalize_img: bool = True,
) -> QM9WithRDKitImages:
    """
    QM9 数据集自动加载函数。
    """
    base_dataset = QM9(root=root, transform=transform, pre_transform=pre_transform)
    return QM9WithRDKitImages(
        base_dataset=base_dataset,
        image_size=image_size,
        add_hs=add_hs,
        normalize_img=normalize_img,
    )


# ============================================================
# 辅助函数：从 PyG Batch 中取出图像 batch
# ============================================================

def extract_image_batch(data: Data) -> torch.Tensor:
    """
    兼容以下几种情况：
    - wrapper/smoke test 推荐格式：data.img = [B, 3, H, W]
    - 单图格式：data.img = [3, H, W]
    - PyG 对 [3, H, W] 默认拼接后的格式：[B*3, H, W]
    """
    if not hasattr(data, "img") or data.img is None:
        raise ValueError("Input data must contain `img` for the RDKit image branch.")

    img = data.img.float()

    if img.dim() == 4:
        return img

    if img.dim() == 3:
        if img.size(0) == 3:
            return img.unsqueeze(0)

        if hasattr(data, "num_graphs") and img.size(0) == data.num_graphs * 3:
            return img.view(data.num_graphs, 3, img.size(-2), img.size(-1))

    raise ValueError(
        f"Unsupported image tensor shape {tuple(img.shape)}. "
        "Expected [B, 3, H, W] or [3, H, W]."
    )


# ============================================================
# 分支A：分子拓扑图编码器
# ============================================================

class MolecularGraphEncoder(nn.Module):
    """
    分支A：分子拓扑图分支。

    严格数据流：
    1. 节点特征 v + 位置编码 Pm -> Linear 投影
    2. Multi-Head Self-Attention，q/k/v 均来自节点表征
    3. Add & Norm
    4. 多层 GCNConv + ReLU
    5. 普通 Conv1d 重构分子节点特征
    6. 多层残差 GATConv 块，动态调整边注意力权重
    7. global_mean_pool -> graph_feat
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        num_gcn_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim % num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads.")

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout = dropout

        # 输入节点特征投影：把 QM9 原始节点特征 + 位置编码 Pm 映射到 hidden_dim。
        self.node_proj = nn.Linear(input_dim, hidden_dim)

        # 1. Multi-Head Self-Attention：batch_first=True，输入 [B, Nmax, hidden_dim]。
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 2. Add & Norm。
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 3. 堆叠多层 GCNConv，每层后接 ReLU。
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)]
        )

        # 4. 普通 Conv 卷积重构分子特征：
        #    将 [B, N, H] 转成 [B, H, N]，对节点序列维做 Conv1d。
        self.reconstruct_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )

        # 5. 残差 GATConv 块。
        # concat=False 保证输出维度仍为 hidden_dim，使 shortcut x + out 合法。
        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
                for _ in range(num_gat_layers)
            ]
        )

        self.latest_mha_attention_weights: Optional[torch.Tensor] = None
        self.latest_gat_attention_weights: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ---------- 输入节点特征 v + 位置编码 Pm ----------
        # x: [num_nodes, input_dim] -> [num_nodes, hidden_dim]
        x = self.node_proj(x.float())

        # ---------- 1. Multi-Head Self-Attention ----------
        # PyTorch MHA 是 dense attention，因此先把稀疏 PyG 节点张量转成 dense batch。
        dense_x, dense_mask = to_dense_batch(x, batch=batch)
        # dense_x: [B, Nmax, H], dense_mask: [B, Nmax], True 表示有效节点。

        attn_out, attn_weights = self.self_attention(
            query=dense_x,
            key=dense_x,
            value=dense_x,
            key_padding_mask=~dense_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        # attn_weights: [B, num_heads, Nmax, Nmax]，保留可解释性分析。
        self.latest_mha_attention_weights = attn_weights

        # ---------- 2. Add & Norm 残差相加 + 层归一化 ----------
        dense_x = self.attn_norm(dense_x + F.dropout(attn_out, p=self.dropout, training=self.training))

        # 从 dense 还原回 PyG 稀疏节点顺序，供 GCN/GAT 使用。
        x = dense_x[dense_mask]

        # ---------- 3. 堆叠多层 GCNConv + ReLU ----------
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)

        # ---------- 4. 普通 Conv1d 卷积重构分子特征 ----------
        dense_x, dense_mask = to_dense_batch(x, batch=batch)
        conv_input = dense_x.transpose(1, 2)               # [B, H, Nmax]
        conv_output = self.reconstruct_conv(conv_input)   # [B, H, Nmax]
        conv_output = F.relu(conv_output).transpose(1, 2) # [B, Nmax, H]
        x = conv_output[dense_mask]

        # ---------- 5. 多层残差 GATConv 块 ----------
        # 每块 = GATConv + ReLU + residual shortcut。
        # GATConv 内部的 alpha 即“动态邻域关联”的可微分边注意力权重。
        self.latest_gat_attention_weights = []

        for gat in self.gat_layers:
            residual = x

            if self.edge_dim is not None and edge_attr is not None:
                gat_out, attn_info = gat(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
            else:
                gat_out, attn_info = gat(
                    x,
                    edge_index,
                    return_attention_weights=True,
                )

            gat_out = F.relu(gat_out)

            # 残差 shortcut：输入直接加到本层输出。
            x = residual + F.dropout(gat_out, p=self.dropout, training=self.training)

            edge_index_with_self_loops, alpha = attn_info
            self.latest_gat_attention_weights.append((edge_index_with_self_loops, alpha))

        # ---------- 6. Global_mean_pool 得到 graph_feat ----------
        graph_feat = global_mean_pool(x, batch)
        return graph_feat

    def get_mha_attention_weights(self) -> Optional[torch.Tensor]:
        """
        返回最近一次 forward 的 MHA 注意力权重。
        """
        return self.latest_mha_attention_weights

    def get_gat_attention_weights(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        返回最近一次 forward 的 GAT 可微分边注意力权重列表。
        每个元素为：
            (edge_index_with_self_loops, alpha)
        """
        return self.latest_gat_attention_weights


# ============================================================
# 分支B：分子 2D 图像编码器
# ============================================================

class MolecularImageEncoder(nn.Module):
    """
    分支B：分子2D图像分支。

    严格数据流：
    1. SMILES 经 RDKit 转 2D 分子图像张量
    2. 堆叠多层 CNN 卷积层
    3. 图像全局均值池化
    4. Linear 投影得到 img_feat
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 多层 CNN 卷积块：Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d。
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 图像全局均值池化：得到每张图像的一维特征。
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, 3, H, W]
        x = self.cnn(img.float())
        x = self.global_avg_pool(x)
        img_feat = self.proj(x)
        return img_feat


# ============================================================
# 融合模块：GATE 自适应门控层
# ============================================================

class GateFusion(nn.Module):
    """
    GATE 自适应门控融合层。

    严格数据流：
    1. graph_feat 与 img_feat 先逐元素相加
    2. sum_feat 经可学习 Linear + sigmoid 得到 gate 权重
    3. gate * graph_feat + (1 - gate) * img_feat
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)
        self.latest_gate_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        graph_feat: torch.Tensor,
        img_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if graph_feat.shape != img_feat.shape:
            raise ValueError(
                f"graph_feat and img_feat must have same shape, got "
                f"{tuple(graph_feat.shape)} vs {tuple(img_feat.shape)}."
            )

        # 1. 先逐元素相加。
        sum_feat = graph_feat + img_feat

        # 2. 可学习 GATE 参数，经 sigmoid 归一化到 [0, 1]。
        gate = torch.sigmoid(self.gate_layer(sum_feat))

        # 3. 自适应调控两支特征融合比例。
        fused = gate * graph_feat + (1.0 - gate) * img_feat

        self.latest_gate_weights = gate
        return fused, gate

    def get_gate_weights(self) -> Optional[torch.Tensor]:
        """
        返回最近一次 forward 的门控融合权重。
        """
        return self.latest_gate_weights


# ============================================================
# 预测头模块：全连接网络 FCN 多任务输出
# ============================================================

class MultiTaskFCNHead(nn.Module):
    """
    多任务 FCN 预测头：
    - 多层隐藏全连接层 h1~hn
    - 每个分子属性一个独立 Linear 输出分支
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        fcn_hidden_dims: Sequence[int] = (256, 128),
        num_tasks: int = 19,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = hidden_dim

        for out_dim in fcn_hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim

        self.shared_fcn = nn.Sequential(*layers)

        # 多分支输出：每个任务独立一层线性层。
        self.task_heads = nn.ModuleList(
            [nn.Linear(in_dim, 1) for _ in range(num_tasks)]
        )

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        h = self.shared_fcn(fused_feat)
        task_outputs = [head(h) for head in self.task_heads]
        return torch.cat(task_outputs, dim=-1)


# ============================================================
# 总模型：KnowledgeAugGAT
# ============================================================

class KnowledgeAugGAT(nn.Module):
    """
    Knowledge-Augmented Graph Attention Network 总模型。

    总架构：
        双独立编码器分支 + GATE 门控融合层 + FCN 多任务预测头

    forward 数据流严格为：
        graph branch -> image branch -> GATE fusion -> FCN head
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_tasks: int = 19,
        num_attention_heads: int = 4,
        num_gcn_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        edge_dim: Optional[int] = None,
        fcn_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.graph_encoder = MolecularGraphEncoder(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_gcn_layers=num_gcn_layers,
            num_gat_layers=num_gat_layers,
            gat_heads=gat_heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.image_encoder = MolecularImageEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.gate_fusion = GateFusion(hidden_dim=hidden_dim)

        self.prediction_head = MultiTaskFCNHead(
            hidden_dim=hidden_dim,
            fcn_hidden_dims=fcn_hidden_dims,
            num_tasks=num_tasks,
            dropout=dropout,
        )

    def forward(
        self,
        data: Data,
        return_explain: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") else None
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        img = extract_image_batch(data).to(x.device)

        # ---------- 分支A：分子拓扑图分支 ----------
        graph_feat = self.graph_encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
        )

        # ---------- 分支B：分子2D图像分支 ----------
        img_feat = self.image_encoder(img)

        # ---------- 融合模块：GATE 自适应门控层 ----------
        fused_feat, gate_weights = self.gate_fusion(graph_feat, img_feat)

        # ---------- 预测头模块：FCN 多任务预测 ----------
        pred = self.prediction_head(fused_feat)

        if not return_explain:
            return pred

        aux = {
            "graph_feat": graph_feat,
            "img_feat": img_feat,
            "fused_feat": fused_feat,
            "mha_attention_weights": self.graph_encoder.get_mha_attention_weights(),
            "gat_attention_weights": self.graph_encoder.get_gat_attention_weights(),
            "gate_weights": gate_weights,
        }
        return pred, aux

    def get_gat_attention_weights(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        提取 GAT 层可微分边注意力权重。
        """
        return self.graph_encoder.get_gat_attention_weights()

    def get_mha_attention_weights(self) -> Optional[torch.Tensor]:
        """
        提取 Multi-Head Self-Attention 权重。
        """
        return self.graph_encoder.get_mha_attention_weights()

    def get_gate_weights(self) -> Optional[torch.Tensor]:
        """
        提取 GATE 门控融合权重。
        """
        return self.gate_fusion.get_gate_weights()


# ============================================================
# 多任务 MSE 损失函数
# ============================================================

class MultiTaskMSELoss(nn.Module):
    """
    QM9 多任务回归 MSE 损失。
    支持可选 mask：
        mask=True 表示该 target 有效
        mask=False 表示缺失或不参与 loss
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'.")
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target.float()

        # PyG QM9 单样本 y 有时为 [1, 19]，Batch 后通常为 [B, 19]；
        # 若出现 [B, 1, 19]，这里压平到与 pred 对齐。
        if target.shape != pred.shape:
            if target.numel() == pred.numel():
                target = target.view_as(pred)
            else:
                raise ValueError(
                    f"Target shape {tuple(target.shape)} cannot match pred shape "
                    f"{tuple(pred.shape)}."
                )

        loss = (pred - target) ** 2

        if mask is not None:
            mask = mask.bool()
            if mask.shape != loss.shape:
                if mask.numel() == loss.numel():
                    mask = mask.view_as(loss)
                else:
                    raise ValueError("Mask shape cannot match loss shape.")
            loss = loss[mask]

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()

        return loss.mean()


# ============================================================
# 训练辅助函数：真实 QM9 数据的最小训练骨架
# ============================================================

def train_one_epoch(
    model: KnowledgeAugGAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskMSELoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(batch)
        loss = criterion(pred, batch.y)

        loss.backward()
        optimizer.step()

        num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else pred.size(0)
        total_loss += float(loss.detach().cpu()) * num_graphs
        total_graphs += num_graphs

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(
    model: KnowledgeAugGAT,
    loader: DataLoader,
    criterion: MultiTaskMSELoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)

        num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else pred.size(0)
        total_loss += float(loss.detach().cpu()) * num_graphs
        total_graphs += num_graphs

    return total_loss / max(total_graphs, 1)


def _batch_smiles(batch: Data) -> List[str]:
    if hasattr(batch, "smiles") and batch.smiles is not None:
        if isinstance(batch.smiles, (list, tuple)):
            return [str(s) for s in batch.smiles]
        if isinstance(batch.smiles, str):
            return [batch.smiles]
    num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
    return [""] * num_graphs


def save_checkpoint(
    path: str,
    model: KnowledgeAugGAT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
    node_input_dim: int,
    edge_dim: Optional[int],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
            "node_input_dim": node_input_dim,
            "edge_dim": edge_dim,
            "target_indices": QM9_HOMO_LUMO_INDICES,
            "target_names": QM9_TARGET_NAMES,
        },
        path,
    )


def append_loss_record(
    csv_path: str,
    epoch: int,
    train_mse: float,
    val_mse: float,
) -> None:
    """
    将每轮 epoch 的训练/验证 MSE 追加写入 CSV。
    文件不存在或为空时自动写入表头。
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    needs_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_mse": f"{train_mse:.10f}",
                "val_mse": f"{val_mse:.10f}",
            }
        )


def init_loss_plot() -> Tuple[plt.Figure, plt.Axes]:
    """
    初始化训练过程实时 loss 曲线窗口。
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Train/Test MSE Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax


def update_loss_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    epochs: Sequence[int],
    train_losses: Sequence[float],
    val_losses: Sequence[float],
) -> None:
    """
    每轮 epoch 后刷新 train_loss 和 val_loss 曲线。
    """
    ax.clear()
    ax.plot(epochs, train_losses, marker="o", label="train_loss")
    ax.plot(epochs, val_losses, marker="s", label="val_loss")
    ax.set_title("Train/Test MSE Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


def save_loss_plot(fig: plt.Figure, output_path: str) -> None:
    """
    保存最终 loss 曲线图片，并保留绘图窗口。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.ioff()
    plt.show(block=False)


def build_model_from_sample(
    sample: Data,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[KnowledgeAugGAT, int, Optional[int]]:
    node_input_dim = sample.x.size(-1)
    if args.use_edge_attr_in_gat and hasattr(sample, "edge_attr") and sample.edge_attr is not None:
        edge_dim = sample.edge_attr.size(-1)
    else:
        edge_dim = None

    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=args.num_tasks,
        num_attention_heads=args.num_attention_heads,
        num_gcn_layers=args.num_gcn_layers,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        edge_dim=edge_dim,
        fcn_hidden_dims=(256, 128),
        dropout=args.dropout,
    ).to(device)
    return model, node_input_dim, edge_dim


def _checkpoint_arg(
    checkpoint_args: Dict[str, object],
    runtime_args: argparse.Namespace,
    name: str,
):
    if name in checkpoint_args and checkpoint_args[name] is not None:
        return checkpoint_args[name]
    return getattr(runtime_args, name)


def build_model_from_checkpoint(
    checkpoint: Dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[KnowledgeAugGAT, int, Optional[int]]:
    """
    根据 checkpoint 元数据构建推理模型，避免预测时手动重复训练超参数。
    """
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing `model_state_dict`.")

    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    node_input_dim = checkpoint.get("node_input_dim")
    if node_input_dim is None:
        state_dict = checkpoint["model_state_dict"]
        node_proj_weight = state_dict.get("graph_encoder.node_proj.weight")
        if node_proj_weight is None:
            raise KeyError("Checkpoint missing `node_input_dim` and graph_encoder.node_proj.weight.")
        node_input_dim = int(node_proj_weight.size(1))
    else:
        node_input_dim = int(node_input_dim)

    edge_dim_value = checkpoint.get("edge_dim", None)
    edge_dim = int(edge_dim_value) if edge_dim_value is not None else None

    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=int(_checkpoint_arg(checkpoint_args, args, "hidden_dim")),
        num_tasks=int(_checkpoint_arg(checkpoint_args, args, "num_tasks")),
        num_attention_heads=int(_checkpoint_arg(checkpoint_args, args, "num_attention_heads")),
        num_gcn_layers=int(_checkpoint_arg(checkpoint_args, args, "num_gcn_layers")),
        num_gat_layers=int(_checkpoint_arg(checkpoint_args, args, "num_gat_layers")),
        gat_heads=int(_checkpoint_arg(checkpoint_args, args, "gat_heads")),
        edge_dim=edge_dim,
        fcn_hidden_dims=(256, 128),
        dropout=float(_checkpoint_arg(checkpoint_args, args, "dropout")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, node_input_dim, edge_dim


@torch.no_grad()
def predict_single_smiles(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float]:
    """
    单 SMILES 推理入口：与训练、测试集评估和 CSV 导出逻辑解耦。
    """
    if not args.smiles:
        raise ValueError("--predict requires --smiles.")

    if not args.checkpoint:
        raise ValueError("--predict requires --checkpoint.")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint!r}.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, node_input_dim, edge_dim = build_model_from_checkpoint(checkpoint, args, device)

    data = smiles_to_inference_data(
        smiles=args.smiles,
        node_input_dim=node_input_dim,
        image_size=args.image_size,
        edge_dim=edge_dim,
        normalize_img=True,
    ).to(device)

    model.eval()
    pred = model(data).view(-1)
    if pred.numel() < 2:
        raise ValueError(
            f"Model output must contain at least HOMO/LUMO predictions, got shape {tuple(pred.shape)}."
        )

    homo = float(pred[0].detach().cpu())
    lumo = float(pred[1].detach().cpu())

    print(f"SMILES: {args.smiles}")
    print(f"HOMO: {homo:.6f}")
    print(f"LUMO: {lumo:.6f}")
    return homo, lumo


@torch.no_grad()
def predict_and_export(
    model: KnowledgeAugGAT,
    loader: DataLoader,
    device: torch.device,
    output_path: str,
    target_names: Sequence[str] = QM9_TARGET_NAMES,
) -> float:
    model.eval()
    rows: List[Dict[str, object]] = []
    total_loss = 0.0
    total_graphs = 0
    criterion = MultiTaskMSELoss(reduction="mean")

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else pred.size(0)
        total_loss += float(loss.detach().cpu()) * num_graphs
        total_graphs += num_graphs

        pred_np = pred.cpu().numpy()
        target_np = batch.y.view(num_graphs, -1).cpu().numpy()
        smiles_list = _batch_smiles(batch)

        for i in range(num_graphs):
            row: Dict[str, object] = {"smiles": smiles_list[i] if i < len(smiles_list) else ""}
            for j, name in enumerate(target_names):
                row[f"pred_{name}"] = float(pred_np[i, j])
                row[f"true_{name}"] = float(target_np[i, j])
                row[f"error_{name}"] = float(pred_np[i, j] - target_np[i, j])
            rows.append(row)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return total_loss / max(total_graphs, 1)


# ============================================================
# 随机样本前向传播测试代码
# ============================================================

def _make_synthetic_molecule_data(
    smiles: str,
    x: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
    image_size: int,
) -> Data:
    """
    构造一个用于 smoke test 的 PyG Data。
    """
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    # 将原始节点特征与位置编码 Pm 拼接，模拟 QM9 输入预处理。
    data.x = build_node_features_with_position_encoding(data)

    # RDKit SMILES -> 2D 图像张量，存成 [1, 3, H, W] 方便 PyG Batch。
    data.img = smiles_to_rdkit_image_tensor(
        smiles,
        image_size=image_size,
        normalize=True,
    ).unsqueeze(0)

    data.smiles = smiles
    return data


def run_smoke_test(
    image_size: int = 128,
    hidden_dim: int = 128,
    num_tasks: int = 2,
    device: Optional[torch.device] = None,
) -> None:
    """
    完整跑通：
    - RDKit 图像分支
    - 图分支 MHA + Add&Norm + GCNConv + Conv1d + 残差 GATConv
    - GATE 融合
    - FCN 多任务预测
    - MultiTaskMSELoss
    - Adam 一次优化
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    np.random.seed(42)

    # 分子1：乙醇 CCO，3个节点，双向边。
    x1 = torch.randn(3, 11)
    pos1 = torch.randn(3, 3)
    edge_index1 = torch.tensor(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]],
        dtype=torch.long,
    )
    edge_attr1 = torch.randn(edge_index1.size(1), 4)
    y1 = torch.randn(1, num_tasks)

    data1 = _make_synthetic_molecule_data(
        smiles="CCO",
        x=x1,
        pos=pos1,
        edge_index=edge_index1,
        edge_attr=edge_attr1,
        y=y1,
        image_size=image_size,
    )

    # 分子2：水 O，单节点，无显式边；GCN/GAT 会自动处理 self-loop。
    x2 = torch.randn(1, 11)
    pos2 = torch.randn(1, 3)
    edge_index2 = torch.empty(2, 0, dtype=torch.long)
    edge_attr2 = torch.empty(0, 4)
    y2 = torch.randn(1, num_tasks)

    data2 = _make_synthetic_molecule_data(
        smiles="O",
        x=x2,
        pos=pos2,
        edge_index=edge_index2,
        edge_attr=edge_attr2,
        y=y2,
        image_size=image_size,
    )

    batch = Batch.from_data_list([data1, data2]).to(device)

    node_input_dim = batch.x.size(-1)
    edge_dim = batch.edge_attr.size(-1) if hasattr(batch, "edge_attr") and batch.edge_attr is not None else None

    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=hidden_dim,
        num_tasks=num_tasks,
        num_attention_heads=4,
        num_gcn_layers=2,
        num_gat_layers=2,
        gat_heads=4,
        edge_dim=edge_dim,
        fcn_hidden_dims=(256, 128),
        dropout=0.1,
    ).to(device)

    criterion = MultiTaskMSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    pred, aux = model(batch, return_explain=True)
    loss = criterion(pred, batch.y)

    loss.backward()
    optimizer.step()

    # ---------- 基础断言：保证完整数据流可运行 ----------
    assert pred.shape == (2, num_tasks), f"Unexpected pred shape: {tuple(pred.shape)}"
    assert torch.isfinite(pred).all(), "Prediction contains NaN or Inf."
    assert torch.isfinite(loss).all(), "Loss contains NaN or Inf."

    gate_weights = aux["gate_weights"]
    assert gate_weights.shape == (2, hidden_dim), f"Unexpected gate shape: {tuple(gate_weights.shape)}"
    assert torch.all(gate_weights >= 0.0) and torch.all(gate_weights <= 1.0), "Gate weights must be in [0, 1]."

    mha_weights = aux["mha_attention_weights"]
    assert mha_weights is not None and mha_weights.numel() > 0, "Missing MHA attention weights."

    gat_weights = aux["gat_attention_weights"]
    assert len(gat_weights) > 0, "Missing GAT attention weights."

    print("Smoke test passed.")
    print(f"Prediction shape: {tuple(pred.shape)}")
    print(f"Loss: {float(loss.detach().cpu()):.6f}")
    print(f"MHA attention shape: {tuple(mha_weights.shape)}")
    print(f"Gate weights shape: {tuple(gate_weights.shape)}")
    for layer_idx, (attn_edge_index, alpha) in enumerate(gat_weights):
        print(
            f"GAT layer {layer_idx}: edge_index shape={tuple(attn_edge_index.shape)}, "
            f"alpha shape={tuple(alpha.shape)}"
        )


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge-Augmented Graph Attention Network for QM9 multi-task regression."
    )

    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic forward/backward test.")
    parser.add_argument("--train", action="store_true", help="Run minimal QM9 training loop.")
    parser.add_argument("--test", action="store_true", help="Evaluate on test split and export predictions.")
    parser.add_argument("--predict", action="store_true", help="Run single-SMILES HOMO/LUMO inference.")
    parser.add_argument("--smiles", type=str, default=None, help="SMILES string for --predict.")

    parser.add_argument("--data-root", type=str, default="./data/QM9", help="QM9 dataset root.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-tasks", type=int, default=2, help="Number of targets (default: 2 for HOMO/LUMO).")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for --test or --predict.")
    parser.add_argument("--output-csv", type=str, default="./results/predictions.csv", help="Prediction export path.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-gcn-layers", type=int, default=2)
    parser.add_argument("--num-gat-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=4)

    parser.add_argument(
        "--use-edge-attr-in-gat",
        action="store_true",
        help="Use QM9 edge_attr as GAT edge features if available.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=1024,
        help="Limit training subset size for quick demo. Set <=0 to use all.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=256,
        help="Limit validation subset size for quick demo. Set <=0 to use all validation split.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=256,
        help="Limit test subset size for quick demo. Set <=0 to use all test split.",
    )

    return parser.parse_args()


def _make_test_loader(args: argparse.Namespace, dataset: QM9WithRDKitImages) -> DataLoader:
    num_total = len(dataset)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    test_indices = list(range(num_train + num_val, num_total))
    if args.max_test_samples > 0:
        test_indices = test_indices[: args.max_test_samples]
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    return DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)


def run_test(args: argparse.Namespace, device: torch.device) -> None:
    checkpoint_path = args.checkpoint or os.path.join(args.save_dir, "best.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path!r}. "
            "Train first with --train or pass --checkpoint."
        )

    dataset = load_qm9_dataset(
        root=args.data_root,
        image_size=args.image_size,
        normalize_img=True,
    )
    test_loader = _make_test_loader(args, dataset)
    sample = dataset[0]

    model, _, _ = build_model_from_sample(sample, args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(
        model=model,
        loader=test_loader,
        criterion=MultiTaskMSELoss(reduction="mean"),
        device=device,
    )
    print(f"test_mse={test_loss:.6f}")

    avg_loss = predict_and_export(
        model=model,
        loader=test_loader,
        device=device,
        output_path=args.output_csv,
    )
    print(f"Predictions exported to {args.output_csv} (mse={avg_loss:.6f})")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.predict:
        try:
            predict_single_smiles(args, device)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise SystemExit(f"Prediction failed: {exc}")
        return

    if args.smoke_test:
        run_smoke_test(
            image_size=args.image_size,
            hidden_dim=args.hidden_dim,
            num_tasks=args.num_tasks,
            device=device,
        )
        if not args.train and not args.test:
            return

    if args.test and not args.train:
        run_test(args, device)
        return

    # 默认行为：若用户没有显式 --train / --test，则跑 smoke test，避免意外下载完整 QM9。
    if not args.train:
        run_smoke_test(
            image_size=args.image_size,
            hidden_dim=args.hidden_dim,
            num_tasks=args.num_tasks,
            device=device,
        )
        return

    dataset = load_qm9_dataset(
        root=args.data_root,
        image_size=args.image_size,
        normalize_img=True,
    )

    # 最小训练/验证切分；为了示例可快速启动，默认限制样本数。
    num_total = len(dataset)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)

    train_indices = list(range(0, num_train))
    val_indices = list(range(num_train, num_train + num_val))

    if args.max_train_samples > 0:
        train_indices = train_indices[:args.max_train_samples]
    if args.max_val_samples > 0:
        val_indices = val_indices[:args.max_val_samples]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = _make_test_loader(args, dataset)

    sample = dataset[0]
    model, node_input_dim, edge_dim = build_model_from_sample(sample, args, device)

    criterion = MultiTaskMSELoss(reduction="mean")

    # Adam 优化器完整配置。
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    results_dir = "./results"
    loss_record_path = os.path.join(results_dir, "loss_record.csv")
    loss_curve_path = os.path.join(results_dir, "loss_curve.png")
    loss_epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    loss_fig, loss_ax = init_loss_plot()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_mse={train_loss:.6f} | "
            f"val_mse={val_loss:.6f}"
        )

        append_loss_record(
            csv_path=loss_record_path,
            epoch=epoch,
            train_mse=train_loss,
            val_mse=val_loss,
        )
        loss_epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        update_loss_plot(
            fig=loss_fig,
            ax=loss_ax,
            epochs=loss_epochs,
            train_losses=train_losses,
            val_losses=val_losses,
        )

        last_path = os.path.join(args.save_dir, "last.pt")
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            args=args,
            node_input_dim=node_input_dim,
            edge_dim=edge_dim,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
                node_input_dim=node_input_dim,
                edge_dim=edge_dim,
            )
            print(f"  saved best checkpoint -> {best_path}")

    save_loss_plot(loss_fig, loss_curve_path)
    print(f"Loss records saved/appended to {loss_record_path}")
    print(f"Loss curve saved to {loss_curve_path}")

    test_loss = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(f"test_mse={test_loss:.6f}")

    avg_loss = predict_and_export(
        model=model,
        loader=test_loader,
        device=device,
        output_path=args.output_csv,
    )
    print(f"Predictions exported to {args.output_csv} (mse={avg_loss:.6f})")

if __name__ == "__main__":
    main()
