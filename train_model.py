import pickle
from torch.utils import data
import time
from torch_geometric.data import Batch
import torch.utils.data.sampler as sampler
import numpy as np
import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.optim as optim
import random
from utils.net_utils import *
from utils.metrics import *
from net.model import GerNA
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
from data_utils.dataset import GerNA_FastDataset, fast_collate_fn

import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp


# ================= 1. 智能日志记录类 =================
class Logger(object):
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)

        # 过滤 tqdm 原地刷新内容，避免日志挤成一行
        if "\r" in message or "%|" in message:
            return

        # 保留正常换行和正常输出
        if message == "\n" or message.strip():
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ================= 2. 随机种子 =================
def set_random_seeds(seed_value=99):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_random_seeds(seed_value=99)


# ================= 3. 安全加载权重函数 =================
def safe_load_state_dict(model, checkpoint_path, device, rank=0):
    """
    安全加载模型权重：
    1. 自动去掉 DDP 保存时产生的 module. 前缀；
    2. 只加载 shape 完全匹配的参数；
    3. 如果专家数、top_k、MBT维度发生变化，不会因为 shape mismatch 直接崩溃。
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    current_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []

    for k, v in checkpoint.items():
        new_k = k[7:] if k.startswith("module.") else k

        if new_k in current_state and current_state[new_k].shape == v.shape:
            filtered_state[new_k] = v
        else:
            skipped_keys.append(new_k)

    current_state.update(filtered_state)
    model.load_state_dict(current_state, strict=True)

    if rank == 0:
        print(f">>> ✅ 成功加载兼容权重数量: {len(filtered_state)}")
        if len(skipped_keys) > 0:
            print(f">>> ⚠️ 跳过不兼容权重数量: {len(skipped_keys)}")
            print(f">>> ⚠️ 示例跳过参数: {skipped_keys[:10]}")

    return model


# ================= 4. 评估函数 =================
def test(net, dataLoader, batch_size, mode, device, threshold=0, desc="Evaluating"):
    output_list = []
    label_list = []

    with torch.no_grad():
        net.eval()

        for batch_index, batch_data in tqdm(
            enumerate(dataLoader),
            total=len(dataLoader),
            desc=desc,
            disable=(mode == "train")
        ):
            batch_data = [
                item.to(device) if (torch.is_tensor(item) or isinstance(item, Batch)) else item
                for item in batch_data
            ]

            [
                b_repre,
                b_seq_mask,
                b_mol_g,
                b_rna_g,
                b_feats,
                b_c4,
                b_coors,
                b_rmask,
                b_mfeats,
                b_mcoors,
                b_mmask,
                b_las,
                b_label
            ] = batch_data

            # 注意：顺序必须是 b_mol_g, b_rna_g
            res = net(
                b_repre,
                b_seq_mask,
                b_mol_g,
                b_rna_g,
                b_feats,
                b_c4,
                b_coors,
                b_rmask,
                b_mfeats,
                b_mcoors,
                b_mmask,
                b_las
            )

            logits = res[0] if isinstance(res, tuple) else res

            if logits.dim() == 1 or logits.shape[-1] == 1:
                probs_pos = torch.sigmoid(logits.reshape(-1))
            else:
                probs = F.softmax(logits, dim=1)
                probs_pos = probs[:, 1]

            batch_probs = np.nan_to_num(
                probs_pos.cpu().detach().numpy(),
                nan=0.5,
                posinf=1.0,
                neginf=0.0
            )

            output_list += batch_probs.tolist()
            label_list += b_label.reshape(-1).cpu().numpy().tolist()

    if mode == "train":
        metrics = get_train_metrics(
            np.array(output_list),
            np.array(label_list)
        )
    else:
        metrics = get_valid_metrics(
            np.array(output_list),
            np.array(label_list),
            threshold
        )

    return metrics, label_list, np.array(output_list)


# ================= 5. 训练与评估主函数 =================
def train_and_eval(
    rank,
    world_size,
    trainDataset,
    trainUnbDataset,
    validDataset,
    testDataset,
    params,
    batch_size=8,
    num_epoch=30,
    model_path=None,
    port=None,
    dataset_name="",
    split_method="",
    log_file=None,
    hparams=None
):
    # 每个子进程重新设置随机种子
    set_random_seeds(seed_value=99 + rank)

    # ---------------- 子进程日志绑定 ----------------
    if rank == 0 and log_file is not None:
        sys.stdout = Logger(log_file)
        sys.stderr = sys.stdout

    if hparams is None:
        hparams = {}

    # ---------------- 分布式初始化 ----------------
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=60)
    )

    # ---------------- 数据加载器 ----------------
    train_sampler = DistributedSampler(
        trainDataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    trainDataLoader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=fast_collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    if rank == 0:
        validDataLoader = torch.utils.data.DataLoader(
            validDataset,
            batch_size=batch_size,
            collate_fn=fast_collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )

        testDataLoader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=batch_size,
            collate_fn=fast_collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )

        train_unb_DataLoader = torch.utils.data.DataLoader(
            trainUnbDataset,
            batch_size=batch_size,
            collate_fn=fast_collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )

        if not os.path.exists(model_path):
            os.makedirs(model_path)

    # ---------------- 超参数读取 ----------------
    lr = float(hparams.get("Learning_Rate", 0.0002))
    wd = float(hparams.get("Weight_Decay", 0.0001))
    acc_steps = int(hparams.get("Accumulation_Steps", 16))

    lambda_res = float(
        hparams.get(
            "Lambda_Resonance",
            hparams.get("Lambda_Res", 0.05)
        )
    )

    lambda_bal = float(
        hparams.get(
            "Lambda_Balance",
            hparams.get("Lambda_Bal", 0.01)
        )
    )

    lambda_geo = float(
        hparams.get(
            "Lambda_Geometric",
            hparams.get("Lambda_Geo", 0.01)
        )
    )

    lambda_dis = float(
        hparams.get(
            "Lambda_Disentangle",
            hparams.get("Lambda_Dis", 0.05)
        )
    )

    num_experts = int(hparams.get("num_experts", 4))
    top_k = int(hparams.get("top_k", 2))
    use_mbt = bool(hparams.get("Use_MBT", True))

    acc_steps = max(acc_steps, 1)

    # ---------------- 保存路径 ----------------
    mbt_tag = "MBT" if use_mbt else "NoMBT"
    save_model_name = f"{dataset_name}_{split_method}-E{num_experts}-T{top_k}-{mbt_tag}.pth"
    save_model_path = os.path.join(model_path, save_model_name)

    if rank == 0:
        print("\n" + "=" * 60)
        print("模型保存路径")
        print(f"  {save_model_path}")
        print("=" * 60 + "\n")

    # ---------------- 模型初始化 ----------------
    net = GerNA(
        params=params,
        trigonometry=True,
        rna_graph=True,
        coors=True,
        coors_3_bead=True,
        uncertainty=True,
        hparams=hparams,
        use_mbt=use_mbt
    ).to(device)

    # ---------------- 初始化或接力加载 ----------------
    if os.path.exists(save_model_path):
        try:
            if rank == 0:
                print(f">>> ✅ 发现历史权重，准备接力训练: {save_model_path}")

            net = safe_load_state_dict(
                model=net,
                checkpoint_path=save_model_path,
                device=device,
                rank=rank
            )

            if rank == 0:
                print(f">>> ✅ 接力成功，权重已载入: {save_model_path}")

        except Exception as e:
            if rank == 0:
                print(f">>> ⚠️ 权重解析失败，将执行随机初始化。错误信息: {e}")

            net.apply(weights_init)

    else:
        if rank == 0:
            print(">>> ✅ 新实验：未发现历史权重，执行随机初始化。")

        net.apply(weights_init)

    # ---------------- DDP 包装 ----------------
    net = DistributedDataParallel(
        net,
        device_ids=[rank] if torch.cuda.is_available() else None,
        output_device=rank if torch.cuda.is_available() else None,
        find_unused_parameters=False
    )

    try:
        net._set_static_graph()
    except Exception:
        if rank == 0:
            print(">>> ⚠️ 当前 PyTorch/DDP 版本不支持 _set_static_graph，已自动跳过。")

    # ---------------- 损失函数与优化器 ----------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        weight_decay=wd,
        amsgrad=True
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.5
    )

    max_auroc = 0.0
    threshold = 0.5

    # ================= 训练主循环 =================
    for epoch in range(num_epoch):
        if rank == 0:
            print(f"\nEpoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.8f}")

        net.train()
        train_sampler.set_epoch(epoch)

        total_loss_tracker = torch.zeros(2, device=device)

        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(
            trainDataLoader,
            desc=f"Epoch {epoch} Training",
            disable=(rank != 0)
        )

        for batch_index, batch_data in enumerate(progress_bar):
            batch_data = [
                item.to(device) if (torch.is_tensor(item) or isinstance(item, Batch)) else item
                for item in batch_data
            ]

            [
                b_repre,
                b_seq_mask,
                b_mol_g,
                b_rna_g,
                b_feats,
                b_c4,
                b_coors,
                b_rmask,
                b_mfeats,
                b_mcoors,
                b_mmask,
                b_las,
                b_label
            ] = batch_data

            labels_long = b_label.reshape(-1).long()
            labels_float = b_label.reshape(-1).float()

            # ---------------- 前向传播 ----------------
            (
                logits,
                alea_unc,
                epis_unc,
                gate_probs,
                pair_2d,
                pair_3d,
                shared_mean,
                spec_2d,
                spec_3d
            ) = net(
                b_repre,
                b_seq_mask,
                b_mol_g,
                b_rna_g,
                b_feats,
                b_c4,
                b_coors,
                b_rmask,
                b_mfeats,
                b_mcoors,
                b_mmask,
                b_las
            )

            # ---------------- A. 分类损失 ----------------
            cls_loss = criterion(
                logits,
                labels_long
            )

            # ---------------- B. MoE 负载均衡损失 ----------------
            if num_experts > 1:
                importance = gate_probs.mean(dim=0)
                balance_loss = (
                    importance.std()
                    / (importance.mean() + 1e-6)
                ).pow(2)
            else:
                balance_loss = torch.tensor(
                    0.0,
                    device=device,
                    dtype=logits.dtype
                )

            # ---------------- C. 2D-3D 几何共振损失 ----------------
            p2d_s = torch.nan_to_num(
                pair_2d,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            p3d_s = torch.nan_to_num(
                pair_3d,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            p3d_aligned = F.adaptive_max_pool2d(
                p3d_s.unsqueeze(1),
                (p2d_s.shape[1], p2d_s.shape[2])
            ).squeeze(1)

            resonance_loss = F.mse_loss(
                p2d_s,
                p3d_aligned
            )

            # ---------------- D. 弱监督几何约束损失 ----------------
            max_c = (
                p2d_s.max(dim=2)[0].max(dim=1)[0]
                + p3d_s.max(dim=2)[0].max(dim=1)[0]
            ) / 2.0

            geo_loss = F.binary_cross_entropy(
                torch.clamp(max_c, 1e-6, 1.0 - 1e-6),
                labels_float
            )

            # ---------------- E. 解耦正交损失 ----------------
            if shared_mean is not None and spec_2d is not None and spec_3d is not None and use_mbt:
                sim_2d = F.cosine_similarity(
                    shared_mean,
                    spec_2d,
                    dim=1
                )

                sim_3d = F.cosine_similarity(
                    shared_mean,
                    spec_3d,
                    dim=1
                )

                disentangle_loss = torch.mean(
                    torch.abs(sim_2d) + torch.abs(sim_3d)
                )
            else:
                disentangle_loss = torch.tensor(
                    0.0,
                    device=device,
                    dtype=logits.dtype
                )

            # ---------------- F. 综合损失 ----------------
            raw_loss = (
                cls_loss
                + lambda_bal * balance_loss
                + lambda_geo * geo_loss
                + lambda_res * resonance_loss
                + lambda_dis * disentangle_loss
            )

            loss = raw_loss / acc_steps

            # ---------------- NaN / Inf 熔断：必须放在 backward 前 ----------------
            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                if rank == 0:
                    print("\n" + "!" * 80)
                    print("❌ [真实崩溃] 检测到 NaN 或 Inf 损失！")
                    print(f"   当前 Epoch: {epoch}")
                    print(f"   当前 Batch: {batch_index}")
                    print(f"   cls_loss        : {cls_loss.item():.6f}")
                    print(f"   balance_loss    : {balance_loss.item():.6f}")
                    print(f"   geo_loss        : {geo_loss.item():.6f}")
                    print(f"   resonance_loss  : {resonance_loss.item():.6f}")
                    print(f"   disentangle_loss: {disentangle_loss.item():.6f}")
                    print("   建议：Learning_Rate 降到 5e-5，Lambda_Resonance 降到 0.01 后重试。")
                    print("!" * 80 + "\n")

                raise ValueError("Exploding gradients: NaN or Inf loss detected.")

            # ---------------- 反向传播 ----------------
            loss.backward()

            # ---------------- 梯度累积更新 ----------------
            is_update_step = ((batch_index + 1) % acc_steps == 0)
            is_last_step = ((batch_index + 1) == len(trainDataLoader))

            if is_update_step or is_last_step:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=1.0
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # ---------------- Loss 统计 ----------------
            batch_num = labels_long.size(0)

            total_loss_tracker[0] += raw_loss.detach() * batch_num
            total_loss_tracker[1] += batch_num

            if rank == 0:
                progress_bar.set_postfix({
                    "loss": f"{raw_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "res": f"{resonance_loss.item():.4f}",
                    "geo": f"{geo_loss.item():.4f}",
                    "dis": f"{disentangle_loss.item():.4f}"
                })

        # ---------------- 同步多卡 Loss ----------------
        dist.all_reduce(
            total_loss_tracker,
            op=dist.ReduceOp.SUM
        )

        avg_epoch_loss = float(
            total_loss_tracker[0] / (total_loss_tracker[1] + 1e-6)
        )

        scheduler.step()

        # ================= Rank 0 评估与保存 =================
        if rank == 0:
            print(f"Epoch {epoch} Avg Loss: {avg_epoch_loss:.4f}")

            perf_name = [
                "TN",
                "FN",
                "FP",
                "TP",
                "Pre",
                "Sen",
                "Spe",
                "Acc",
                "F1",
                "Mcc",
                "AUC",
                "AUPRC"
            ]

            # -------- 验证集评估 --------
            v_perf, _, _ = test(
                net.module,
                validDataLoader,
                batch_size,
                "valid",
                device,
                threshold,
                desc="Valid"
            )

            print(
                "Valid Performance: "
                + " ".join([
                    f"{perf_name[i]}:{v_perf[i]:.4f}"
                    for i in range(len(perf_name))
                ])
            )

            # -------- 保存最佳权重 --------
            if v_perf[-2] > max_auroc:
                max_auroc = v_perf[-2]

                torch.save(
                    net.module.state_dict(),
                    save_model_path
                )

                print(f">>> ✅ 发现新高 Valid AUC: {max_auroc:.4f}，模型已保存。")

            # -------- 测试集评估 --------
            t_perf, _, _ = test(
                net.module,
                testDataLoader,
                batch_size,
                "test",
                device,
                threshold,
                desc="Test"
            )

            print(
                "Test Performance:  "
                + " ".join([
                    f"{perf_name[i]}:{t_perf[i]:.4f}"
                    for i in range(len(perf_name))
                ])
            )

        dist.barrier()

    if rank == 0:
        print("\n✅ 训练任务已圆满结束。")

    dist.destroy_process_group()


# ================= 6. 入口函数 =================
if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="GerNA-Cert Hyperparameter Tracker"
    )

    # ---------------- 基础命令行参数 ----------------
    parser.add_argument(
        "--dataset",
        type=str,
        default="Robin"
    )

    parser.add_argument(
        "--split_method",
        type=str,
        default="random"
    )

    parser.add_argument(
        "--model_output_path",
        type=str,
        default="Model/"
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=120
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--cuda",
        type=str,
        default="2,3"
    )

    parser.add_argument(
        "--port",
        type=str,
        default="12333"
    )

    # ---------------- 架构参数 ----------------
    parser.add_argument(
        "--hidden_size1",
        type=int,
        default=128
    )

    parser.add_argument(
        "--hidden_size2",
        type=int,
        default=128
    )

    # 论文主模型建议 default=4；
    # 如果要跑 E8-T2 实验，
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=2
    )

    # 是否使用 MBT / DBT
    # 1 表示使用；0 表示关闭，用于 w/o MBT 消融实验
    parser.add_argument(
        "--use_mbt",
        type=int,
        default=1
    )

    args = parser.parse_args()

    # ================= 7. 手动定义的隐藏超参数 =================
    hparams = {
        # ===== Optimizer / Training =====
        "Learning_Rate": 0.0002,
        "Weight_Decay": 0.0001,
        "Accumulation_Steps": 16,

        # ===== DBT / MBT module =====
        "MBT_Num_Bottlenecks": 8,
        "MBT_Proj_Dim": 512,

        # 这里两个 key 都保留：
        # Residual_Alpha 给论文描述用；
        # MBT_Residual_Alpha 给 model.py 兼容读取用。
        "Residual_Alpha": 0.5,
        "MBT_Residual_Alpha": 0.5,

        "Dropout_Rate": 0.5,

        # ===== Loss weights =====
        "Lambda_Resonance": 0.05,
        "Lambda_Balance": 0.01,
        "Lambda_Geometric": 0.01,
        "Lambda_Disentangle": 0.05,

        # ===== MoE predictor =====
        "num_experts": 4,
        "top_k": 1,

        # ===== Ablation switch =====
        "Use_MBT": bool(args.use_mbt),

        # ===== Sampling =====
        "Positive_Oversampling": 10
    }

    # ================= 8. 日志初始化 =================
    log_dir = "Outputs_Logs"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    mbt_tag = "MBT" if hparams["Use_MBT"] else "NoMBT"

    log_file_path = os.path.join(
        log_dir,
        f"{args.dataset}_{args.split_method}_E{args.num_experts}_T{args.top_k}_{mbt_tag}_{datetime.now().strftime('%m%d_%H%M%S')}.log"
    )

    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    print("\n" + "=" * 60)
    print(f"GerNA-Cert 实验启动 | 任务: {args.dataset} ({args.split_method})")
    print("=" * 60)

    print("\n[Section 1: Command Line Args]")
    for arg in vars(args):
        print(f"  {arg:<24}: {getattr(args, arg)}")

    print("\n[Section 2: Training & Loss Hyperparameters]")
    for key, value in hparams.items():
        print(f"  {key:<24}: {value}")

    print("\n[Section 3: System Info]")
    print(f"  CUDA_VISIBLE_DEVICES    : {args.cuda}")
    print(f"  Log_File_Path           : {log_file_path}")
    print("=" * 60 + "\n")

    # ================= 9. CUDA 设置 =================
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有检测到 CUDA。你的 DDP 训练脚本需要 GPU。")

    world_size = torch.cuda.device_count()

    if world_size < 1:
        raise RuntimeError("没有可用 GPU，请检查 CUDA_VISIBLE_DEVICES 设置。")

    print(f">>> 检测到可用 GPU 数量: {world_size}")

    # ================= 10. 数据路径 =================
    train_pt = f"./data/{args.dataset}/{args.split_method}/train_pt"
    valid_pt = f"./data/{args.dataset}/{args.split_method}/valid_pt"
    test_pt = f"./data/{args.dataset}/{args.split_method}/test_pt"

    train_pkl_path = f"./data/{args.dataset}/{args.split_method}/train_data.pkl"

    if not os.path.exists(train_pt):
        raise FileNotFoundError(f"训练集路径不存在: {train_pt}")

    if not os.path.exists(valid_pt):
        raise FileNotFoundError(f"验证集路径不存在: {valid_pt}")

    if not os.path.exists(test_pt):
        raise FileNotFoundError(f"测试集路径不存在: {test_pt}")

    if not os.path.exists(train_pkl_path):
        raise FileNotFoundError(f"训练标签文件不存在: {train_pkl_path}")

    # ================= 11. 数据集读取 =================
    trainUnbDataset = GerNA_FastDataset(train_pt)
    validDataset = GerNA_FastDataset(valid_pt)
    testDataset = GerNA_FastDataset(test_pt)

    print(f">>> 原始训练集数量: {len(trainUnbDataset)}")
    print(f">>> 验证集数量    : {len(validDataset)}")
    print(f">>> 测试集数量    : {len(testDataset)}")

    # ================= 12. 正样本过采样 =================
    with open(train_pkl_path, "rb") as f:
        label = pickle.load(f)[-1]

    train_idx = list(range(len(trainUnbDataset)))

    pos_count = 0
    neg_count = 0

    for i in range(len(label)):
        if int(label[i]) == 1:
            pos_count += 1
            train_idx.extend([i] * int(hparams["Positive_Oversampling"]))
        else:
            neg_count += 1

    trainDataset = data.Subset(
        trainUnbDataset,
        train_idx
    )

    print(f">>> 原始正样本数量: {pos_count}")
    print(f">>> 原始负样本数量: {neg_count}")
    print(f">>> 过采样倍数    : {hparams['Positive_Oversampling']}")
    print(f">>> 过采样后训练集: {len(trainDataset)}")

    # ================= 13. 模型结构参数 =================
    params = [
        4,
        2,
        args.hidden_size1,
        args.hidden_size2
    ]

    print("\n[Section 4: Model Params]")
    print(f"  GNN_depth          : {params[0]}")
    print(f"  DMA_depth          : {params[1]}")
    print(f"  hidden_size1       : {params[2]}")
    print(f"  hidden_size2       : {params[3]}")
    print(f"  num_experts        : {hparams['num_experts']}")
    print(f"  top_k              : {hparams['top_k']}")
    print(f"  Use_MBT            : {hparams['Use_MBT']}")
    print("=" * 60 + "\n")

    # ================= 14. 启动 DDP 训练 =================
    mp.spawn(
        train_and_eval,
        args=(
            world_size,
            trainDataset,
            trainUnbDataset,
            validDataset,
            testDataset,
            params,
            args.batch_size,
            args.epoch,
            args.model_output_path,
            args.port,
            args.dataset,
            args.split_method,
            log_file_path,
            hparams
        ),
        nprocs=world_size,
        join=True
    )

    total_time = timedelta(
        seconds=int(time.time() - start_time)
    )

    print(f"\n✅ 实验结束！总耗时: {total_time}")
