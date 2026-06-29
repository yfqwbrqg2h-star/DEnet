"""QM9数据集工具（节点特征、数据集包装、图像batch提取）"""
import copy
import glob
import os
from typing import Optional, Dict, List

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import QM9

from .smiles_processing import smiles_to_rdkit_image_tensor


def _one_hot_atomic_number(z: torch.Tensor, max_atomic_num: int = 100) -> torch.Tensor:
    """当Data.x不存在时，用原子序数构造one-hot节点特征"""
    z = z.long().clamp(min=0, max=max_atomic_num)
    return torch.nn.functional.one_hot(z, num_classes=max_atomic_num + 1).float()


def _centered_position_encoding(
    pos: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """构造位置编码Pm：中心化坐标 + 径向距离"""
    pos = pos.float()

    if batch is None:
        centroid = pos.mean(dim=0, keepdim=True)
        centered = pos - centroid
    else:
        batch = batch.long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        counts = torch.bincount(batch, minlength=num_graphs).float().clamp_min(1.0)
        centroid_sum = torch.zeros(num_graphs, pos.size(-1), device=pos.device, dtype=pos.dtype)
        centroid_sum.index_add_(0, batch, pos)
        centroids = centroid_sum / counts.unsqueeze(-1)
        centered = pos - centroids[batch]

    radial = torch.norm(centered, p=2, dim=-1, keepdim=True)
    return torch.cat([centered, radial], dim=-1)


def build_node_features_with_position_encoding(
    data: Data,
    max_atomic_num: int = 100,
) -> torch.Tensor:
    """构建QM9节点特征（原始特征 + 位置编码Pm）"""
    # 基础节点特征
    if hasattr(data, "x") and data.x is not None:
        base_x = data.x.float()
    elif hasattr(data, "z") and data.z is not None:
        base_x = _one_hot_atomic_number(data.z, max_atomic_num=max_atomic_num)
    else:
        raise ValueError("Data must contain `x` or `z` for node features.")

    # 位置编码
    if hasattr(data, "pos") and data.pos is not None:
        batch = data.batch if hasattr(data, "batch") else None
        pos_enc = _centered_position_encoding(data.pos, batch=batch).to(base_x.device)
    else:
        pos_enc = torch.zeros(base_x.size(0), 4, dtype=base_x.dtype, device=base_x.device)

    return torch.cat([base_x, pos_enc], dim=-1)


def extract_image_batch(data: Data) -> torch.Tensor:
    """从PyG Batch中提取图像batch（兼容多种格式）"""
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


class QM9WithRDKitImages(Dataset):
    """包装PyG QM9：自动添加节点位置编码 + RDKit 2D图像特征"""
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
        """从Data或原始SDF中提取SMILES"""
        # 优先从data属性中取
        for attr_name in ("smiles", "smile", "SMILES"):
            value = getattr(data, attr_name, None)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str):
                return value[0]

        # 从SDF缓存中取
        if self._sdf_smiles_cache is None:
            self._sdf_smiles_cache = self._read_smiles_from_raw_sdf()

        raw_idx = int(getattr(data, "idx", dataset_index))
        if raw_idx < 0 or raw_idx >= len(self._sdf_smiles_cache):
            raise IndexError(f"Invalid raw idx {raw_idx} for SMILES cache")
        smiles = self._sdf_smiles_cache[raw_idx]
        if not smiles:
            raise ValueError(f"Empty SMILES at raw idx {raw_idx}")
        return smiles

    def _read_smiles_from_raw_sdf(self) -> List[str]:
        """从QM9原始SDF中读取SMILES"""
        raw_dir = getattr(self.base_dataset, "raw_dir", None)
        if raw_dir is None:
            raise RuntimeError("Cannot locate QM9 raw_dir")

        # 找到gdb9.sdf文件
        sdf_candidates = sorted(glob.glob(os.path.join(raw_dir, "*.sdf")))
        if not sdf_candidates:
            raise FileNotFoundError(f"No .sdf in {raw_dir}")
        
        sdf_path = next((p for p in sdf_candidates if "gdb9" in os.path.basename(p).lower()), sdf_candidates[0])
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)

        smiles_list = []
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

        # 添加节点位置编码
        data.x = build_node_features_with_position_encoding(data, max_atomic_num=self.max_atomic_num)

        # 添加图像特征
        smiles = self._extract_smiles_from_data(data, dataset_index=index)
        if smiles not in self._image_cache:
            self._image_cache[smiles] = smiles_to_rdkit_image_tensor(
                smiles, image_size=self.image_size, add_hs=self.add_hs, normalize=self.normalize_img
            )
        data.img = self._image_cache[smiles].unsqueeze(0)
        data.smiles = smiles

        # 只保留HOMO/LUMO标签
        from knowledge_aug_gat.constants import QM9_HOMO_LUMO_INDICES
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
    """加载带图像特征的QM9数据集"""
    base_dataset = QM9(root=root, transform=transform, pre_transform=pre_transform)
    return QM9WithRDKitImages(
        base_dataset=base_dataset,
        image_size=image_size,
        add_hs=add_hs,
        normalize_img=normalize_img,
    )
