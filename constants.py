"""常量定义（QM9标签、可视化配置）"""
# QM9 标签索引：homo=2, lumo=3（PyG QM9 标准顺序）
QM9_HOMO_LUMO_INDICES = [2, 3]
QM9_TARGET_NAMES = ("homo", "lumo")

# Matplotlib 中文和负号兼容配置，适配 Windows 中文环境。
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
