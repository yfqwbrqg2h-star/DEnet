"""主程序（训练/测试/推理）"""
import argparse
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from knowledge_aug_gat import (
    QM9_HOMO_LUMO_INDICES, QM9_TARGET_NAMES,
    MultiTaskMSELoss, load_qm9_dataset, KnowledgeAugGAT,
    smiles_to_inference_data
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Knowledge-Augmented GAT for QM9")
    # 通用参数
    parser.add_argument("--data-root", default="./data/QM9", help="QM9数据集根目录")
    parser.add_argument("--image-size", type=int, default=128, help="分子图像尺寸")
    parser.add_argument("--hidden-dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout概率")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--checkpoint", default=None, help="模型检查点路径")

    # 模式选择
    parser.add_argument("--smoke-test", action="store_true", help="快速前向/反向传播测试")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--predict", action="store_true", help="单SMILES推理模式")
    parser.add_argument("--smiles", default="CCO", help="推理用SMILES")

    # 模型参数
    parser.add_argument("--num-attention-heads", type=int, default=4, help="MHA头数")
    parser.add_argument("--num-gcn-layers", type=int, default=2, help="GCN层数")
    parser.add_argument("--num-gat-layers", type=int, default=2, help="GAT层数")
    parser.add_argument("--gat-heads", type=int, default=4, help="GAT头数")

    return parser.parse_args()


def smoke_test(args):
    """快速前向/反向传播测试"""
    print("=== 烟雾测试：前向+反向传播 ===")
    # 加载少量数据
    dataset = load_qm9_dataset(root=args.data_root, image_size=args.image_size)
    dataloader = DataLoader(dataset[:10], batch_size=2, shuffle=True)
    node_input_dim = dataset[0].x.size(-1)

    # 初始化模型
    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=len(QM9_HOMO_LUMO_INDICES),
        num_attention_heads=args.num_attention_heads,
        num_gcn_layers=args.num_gcn_layers,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(args.device)

    # 损失函数 + 优化器
    criterion = MultiTaskMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 前向+反向
    model.train()
    for batch in dataloader:
        batch = batch.to(args.device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Smoke test loss: {loss.item():.4f}")
        break
    print("=== 烟雾测试完成 ===")


def train(args):
    """训练模型"""
    print("=== 开始训练 ===")
    # 加载数据集
    dataset = load_qm9_dataset(root=args.data_root, image_size=args.image_size)
    train_dataset = dataset[:10000]  # 简化：取前10000个样本
    val_dataset = dataset[10000:11000]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    node_input_dim = dataset[0].x.size(-1)

    # 初始化模型/损失/优化器
    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=len(QM9_HOMO_LUMO_INDICES),
        num_attention_heads=args.num_attention_heads,
        num_gcn_layers=args.num_gcn_layers,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(args.device)

    criterion = MultiTaskMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 加载检查点
    best_val_loss = float("inf")
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val_loss = ckpt["val_loss"]
        print(f"加载检查点：最佳验证损失 {best_val_loss:.4f}")

    # 训练循环
    os.makedirs("./checkpoints", exist_ok=True)
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)"):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Val)"):
                batch = batch.to(args.device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs

        # 平均损失
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "epoch": epoch,
            }, "./checkpoints/best.pt")
            print(f"保存最佳模型：验证损失 {best_val_loss:.4f}")

    print("=== 训练完成 ===")


def test(args):
    """测试模型"""
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise ValueError("测试模式必须指定有效检查点路径")

    print("=== 开始测试 ===")
    dataset = load_qm9_dataset(root=args.data_root, image_size=args.image_size)
    test_dataset = dataset[11000:12000]  # 简化：取1000个测试样本
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    node_input_dim = dataset[0].x.size(-1)

    # 加载模型
    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=len(QM9_HOMO_LUMO_INDICES),
        num_attention_heads=args.num_attention_heads,
        num_gcn_layers=args.num_gcn_layers,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    criterion = MultiTaskMSELoss()

    # 测试
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(args.device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            test_loss += loss.item() * batch.num_graphs

    test_loss /= len(test_dataset)
    print(f"测试平均损失：{test_loss:.4f}")
    print("=== 测试完成 ===")


def predict(args):
    """单SMILES推理"""
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise ValueError("推理模式必须指定有效检查点路径")

    print(f"=== 推理 SMILES: {args.smiles} ===")
    # 加载模型（需要先获取node_input_dim，这里简化为105=101+4）
    node_input_dim = 105  # QM9节点特征(101) + 位置编码(4)
    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=len(QM9_HOMO_LUMO_INDICES),
        num_attention_heads=args.num_attention_heads,
        num_gcn_layers=args.num_gcn_layers,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(args.device)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])

    # 构建推理数据
    data = smiles_to_inference_data(
        smiles=args.smiles,
        node_input_dim=node_input_dim,
        image_size=args.image_size,
    ).to(args.device)

    # 推理
    model.eval()
    with torch.no_grad():
        pred = model(data)
        pred = pred.cpu().numpy().flatten()

    # 输出结果
    for name, value in zip(QM9_TARGET_NAMES, pred):
        print(f"{name.upper()}: {value:.4f}")
    print("=== 推理完成 ===")


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False  # 禁用cudnn加速（避免兼容性问题）

    if args.smoke_test:
        smoke_test(args)
    elif args.train:
        train(args)
    elif args.test:
        test(args)
    elif args.predict:
        predict(args)
    else:
        print("请指定运行模式：--smoke-test / --train / --test / --predict")


if __name__ == "__main__":
    main()
