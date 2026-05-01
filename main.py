import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict
import pickle
import os
import gc
import ray
import matplotlib.pyplot as plt

# 导入自定义模块
from models import CNN, MnistCNN, ResNet18, ResNet20
from client import FlowerClient, load_datasets
from server import PruningFedAvg, evaluate_global_model
from pruning_utils import get_parameters_as_vector, set_parameters_from_vector
import recorder

RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records")
recorder.set_records_dir(RECORDS_DIR)

# ==========================================
# 实验预设 (一键切换)
#   HDM_RESNET20   : 原始配置 (ResNet-20, 200 rounds, alpha=1.0)
#   FEDRTS_RESNET18: 对齐 FedMef (CVPR'24) / FedRTS (NeurIPS'25) 的 CIFAR-10 + ResNet-18 设置
# ==========================================
EXPERIMENT_PRESET = "FEDRTS_RESNET18"

if EXPERIMENT_PRESET == "FEDRTS_RESNET18":
    NUM_CLIENTS = 100
    NUM_ROUNDS = 500          # FedMef R=500, FedRTS T=500
    FRACTION_FIT = 0.1        # 每轮 10/100 客户端 (FedRTS K=10)
    FRACTION_EVALUATE = 0.1
    BATCH_SIZE = 64           # FedRTS: batch 64
    CLIENT_EPOCHS = 5         # FedRTS: 5 local epochs
    LEARNING_RATE = 0.01      # FedRTS: SGD lr=0.01
    CLIENT_MOMENTUM = 0.9
    DATASET_NAME = "CIFAR10"
    MODEL_NAME = "ResNet18"
    IID = False
    ALPHA = 0.5               # FedMef/FedRTS: Dirichlet alpha=0.5
    CLIENT_NUM_GPUS = 0.1     # ResNet-18 参数量约 ResNet-20 的 43 倍, 降低并发
else:  # HDM_RESNET20 (原始配置)
    NUM_CLIENTS = 100
    NUM_ROUNDS = 200
    FRACTION_FIT = 0.2
    FRACTION_EVALUATE = 0.2
    BATCH_SIZE = 128
    CLIENT_EPOCHS = 2
    LEARNING_RATE = 0.005
    CLIENT_MOMENTUM = 0.9
    DATASET_NAME = "CIFAR10"
    MODEL_NAME = "ResNet20"
    IID = False
    ALPHA = 1.0
    CLIENT_NUM_GPUS = 0.05

PRUNING_STRATEGY = "hdm"  # 'l1', 'random', 'hdm', 'mp', 'mag'

# 只跑指定的实验组 (空列表 = 全部跑). 例如: ["Magnitude"] 表示只跑 Magnitude.
RUN_ONLY = ["HDM"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Preset: {EXPERIMENT_PRESET} | Model: {MODEL_NAME} | Rounds: {NUM_ROUNDS} | alpha: {ALPHA} | K: {int(NUM_CLIENTS*FRACTION_FIT)}")

# ==========================================
# 2. 定义四个实验组 (对照实验)
# ==========================================
EXPERIMENTS_CONFIG = [
    {
        "name": "None",
        "strategy": "none",
        "threshold": 0.0,
        "color": "tab:blue"
    },
    {
        "name": "Magnitude",
        "strategy": "magnitude",
        "threshold": 0.70,
        "color": "tab:orange"
    },
    {
        "name": "Delta",
        "strategy": "delta_prun_mad",
        "threshold": 0.6,
        "color": "tab:green"
    },
    {
        "name": "HDM",
        "strategy": "hdm_prun",
        "threshold": 0.92,
        "delta_mad_scale": 2.5,
        "server_momentum_beta": 0.0,
        "color": "tab:purple"
    }
]

# 存储实验结果
all_results = {}

# 1. 准备数据
print(f"Loading {DATASET_NAME} dataset...")
client_datasets, testset = load_datasets(DATASET_NAME, NUM_CLIENTS, iid=IID, alpha=ALPHA)

# 2. 初始化全局模型 (用于获取初始参数形状)
print("Initializing global model...")
if DATASET_NAME == "CIFAR100":
    num_classes = 100
else:
    num_classes = 10

if MODEL_NAME == "ResNet18":
    initial_model = ResNet18(num_classes=num_classes)
elif MODEL_NAME == "ResNet20":
    initial_model = ResNet20(num_classes=num_classes)
elif DATASET_NAME == "MNIST":
    initial_model = MnistCNN(num_classes=num_classes)
else:
    initial_model = CNN(num_classes=num_classes)

initial_model.to(DEVICE)
initial_global_weights_vector = get_parameters_as_vector(initial_model)


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# ==========================================
# 3. 循环运行实验
# ==========================================
ACTIVE_EXPERIMENTS = [e for e in EXPERIMENTS_CONFIG if (not RUN_ONLY) or e["name"] in RUN_ONLY]
print(f"Active experiments: {[e['name'] for e in ACTIVE_EXPERIMENTS]}")
for exp in ACTIVE_EXPERIMENTS:
    exp_name = exp["name"]
    strategy_name = exp["strategy"]
    threshold_val = exp["threshold"]
    delta_mad_scale_val = exp.get("delta_mad_scale", 2.0)

    if strategy_name in ("hdm_prun", "delta_prun_mad"):
        print(f"\n>>> Running Experiment: {exp_name} (Strategy: {strategy_name}, Threshold: {threshold_val}, Delta MAD scale: {delta_mad_scale_val})")
    else:
        print(f"\n>>> Running Experiment: {exp_name} (Strategy: {strategy_name}, Threshold: {threshold_val})")
    recorder.start_experiment(exp_name)


    # 定义Client工厂 (闭包传参)
    def make_client_fn(client_datasets_list, device, s_name, t_val, d_scale):
        def client_fn_decorated(cid: str) -> fl.client.Client:
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # 获取数据
            client_dataset = client_datasets_list[int(cid)]
            trainloader, valloader = None, None
            if client_dataset and len(client_dataset) > 0:
                train_size = int(0.8 * len(client_dataset))
                val_size = len(client_dataset) - train_size
                if val_size == 0:
                    train_data, val_data = client_dataset, None
                else:
                    train_data, val_data = random_split(client_dataset, [train_size, val_size])
                trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
                if val_data: valloader = DataLoader(val_data, batch_size=BATCH_SIZE)

            # 创建模型
            if MODEL_NAME == "ResNet18":
                net = ResNet18(num_classes=num_classes)
            elif MODEL_NAME == "ResNet20":
                net = ResNet20(num_classes=num_classes)
            elif DATASET_NAME == "MNIST":
                net = MnistCNN(num_classes=num_classes)
            else:
                net = CNN(num_classes=num_classes)
            net.to(device)
            set_parameters_from_vector(net, initial_global_weights_vector)

            # 创建Client并注入配置
            client = FlowerClient(cid, net, trainloader, valloader, device)
            client.static_config = {"pruning_strategy": s_name, "threshold_param": t_val, "delta_mad_scale": d_scale}
            return client.to_client()

        return client_fn_decorated


    # 定义策略
    strategy = PruningFedAvg(
        pruning_strategy=strategy_name,
        threshold_param=threshold_val,
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(initial_model)),
        fraction_fit=FRACTION_FIT,  # 每轮选10个客户端
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=max(2, int(NUM_CLIENTS * FRACTION_FIT)),
        min_evaluate_clients=max(2, int(NUM_CLIENTS * FRACTION_EVALUATE)),
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=lambda server_round, parameters, config: evaluate_global_model(server_round, parameters,
                                                                               {"dataset_name": DATASET_NAME,
                                                                                "model_name": MODEL_NAME,
                                                                                "batch_size": BATCH_SIZE}),
        on_fit_config_fn=lambda server_round: {
            "epochs": CLIENT_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "client_momentum": CLIENT_MOMENTUM,
            "dataset_name": DATASET_NAME, "server_round": server_round,
            "pruning_strategy": strategy_name, "threshold_param": threshold_val,
            "delta_mad_scale": delta_mad_scale_val
        },
        strategy_config={
            "server_momentum_beta": exp.get("server_momentum_beta", 0.0),
        },
    )

    # 启动模拟
    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(client_datasets, DEVICE, strategy_name, threshold_val, delta_mad_scale_val),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": CLIENT_NUM_GPUS},  # 4090D 资源配置
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )

    # 保存结果
    all_results[exp_name] = {
        "global_accuracies": history.metrics_centralized.get("accuracy", []),
        "client_metrics": strategy.metrics_aggregated_per_round
    }

    # 实验结束后 dump 完整 pickle, 便于后续重绘图不用重跑
    pkl_path = os.path.join(RECORDS_DIR, f"{exp_name}_full.pkl")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "exp_name": exp_name,
                "strategy": strategy_name,
                "threshold": threshold_val,
                "model": MODEL_NAME,
                "dataset": DATASET_NAME,
                "num_rounds": NUM_ROUNDS,
                "alpha": ALPHA,
                "fraction_fit": FRACTION_FIT,
                "global_accuracies": list(history.metrics_centralized.get("accuracy", [])),
                "client_metrics": list(strategy.metrics_aggregated_per_round),
            }, f)
        print(f"Saved full pickle: {pkl_path}")
    except Exception as e:
        print(f"Warning: failed to dump pickle for {exp_name}: {e}")

    ray.shutdown()
    torch.cuda.empty_cache()
    gc.collect()


# ==========================================
# 4. 自动绘图函数 (一鱼三吃)
# ==========================================
def generate_plot(title, filename, strategies_to_show, max_round):
    # 只保留实际已经跑过的策略
    strategies_to_show = [s for s in strategies_to_show if s in all_results]
    if not strategies_to_show:
        print(f"Skip plot {filename}: no available results.")
        return

    # 单实验 vs 多实验: 单实验时标题改用具体策略名, 不用 "Avg/Global" 这种对比口吻
    is_single = len(strategies_to_show) == 1
    only_name = strategies_to_show[0] if is_single else None

    print(f"Generating plot: {filename}...")
    plt.figure(figsize=(20, 6))

    # 子图1: Accuracy
    plt.subplot(1, 3, 1)
    for exp_conf in EXPERIMENTS_CONFIG:
        name = exp_conf["name"]
        if name not in strategies_to_show: continue

        data = all_results[name]
        rounds, accs = zip(*data["global_accuracies"])
        rounds = [r for r in rounds if r <= max_round]
        accs = accs[:len(rounds)]
        plt.plot(rounds, accs, label=name, color=exp_conf["color"], linewidth=2)

    plt.title(f"{only_name} Accuracy" if is_single else "Global Accuracy")
    plt.xlabel("Rounds"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()

    # 子图2: Upload Size
    plt.subplot(1, 3, 2)
    for exp_conf in EXPERIMENTS_CONFIG:
        name = exp_conf["name"]
        if name not in strategies_to_show: continue

        data = all_results[name]
        metrics = data["client_metrics"]
        rounds = [m["round"] for m in metrics if m["round"] <= max_round]
        uploads = [m["avg_upload_size_kb"] for m in metrics][:len(rounds)]
        plt.plot(rounds, uploads, label=name, color=exp_conf["color"], linewidth=2)

    plt.title(f"{only_name} Upload Size per Client (KB)" if is_single else "Avg Upload Size (KB)")
    plt.xlabel("Rounds"); plt.grid(True); plt.legend()

    # 子图3: Fit Duration
    plt.subplot(1, 3, 3)
    for exp_conf in EXPERIMENTS_CONFIG:
        name = exp_conf["name"]
        if name not in strategies_to_show: continue

        data = all_results[name]
        metrics = data["client_metrics"]
        rounds = [m["round"] for m in metrics if m["round"] <= max_round]
        times = [m["avg_fit_duration"] for m in metrics][:len(rounds)]
        plt.plot(rounds, times, label=name, color=exp_conf["color"], linewidth=2)

    plt.title(f"{only_name} Fit Duration per Client (s)" if is_single else "Avg Fit Duration (s)")
    plt.xlabel("Rounds"); plt.grid(True); plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show() # 服务器上运行时注释掉


# ==========================================
# 5. 生成三张图
# ==========================================

# 实验1：验证短轮次 Delta 优势 (前5轮)
# generate_plot(
#     title="Exp 1: Short-term Efficiency (Round 0-50)",
#     filename="exp1_short_term.png",
#     strategies_to_show=["None", "Magnitude", "Delta"],
#     max_round=50
# )
#
# # 实验2：验证长轮次 Delta 崩塌 (前20-25轮)
# generate_plot(
#     title="Exp 2: Convergence Issue (Round 0-200)",
#     filename="exp2_convergence_collapse.png",
#     strategies_to_show=["None", "Magnitude", "Delta"],
#     max_round=200
# )

# 根据本次实际跑了什么, 自动决定出图
ACTIVE_NAMES = [e["name"] for e in ACTIVE_EXPERIMENTS]
if len(ACTIVE_NAMES) == 1:
    only = ACTIVE_NAMES[0]
    generate_plot(
        title=f"{only} Single Run (Round 0-{NUM_ROUNDS})",
        filename=f"exp_{only.lower()}_single.png",
        strategies_to_show=[only],
        max_round=NUM_ROUNDS,
    )
else:
    generate_plot(
        title=f"Pruning Strategy Comparison (Round 0-{NUM_ROUNDS})",
        filename="exp_comparison.png",
        strategies_to_show=ACTIVE_NAMES,
        max_round=NUM_ROUNDS,
    )

print("\nAll experiments finished. Plots saved.")

# ==========================================
# 6. 输出统计结果到txt
# ==========================================
def save_summary_to_txt(filename="结果.txt"):
    print(f"Saving summary to {filename}...")

    # 计算模型总参数量 (Dense)
    total_params = sum(p.numel() for p in initial_model.parameters())

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"实验结果汇总 (Experiment Summary)\n")
        f.write(f"========================================\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Total Dense Params: {total_params:,}\n")
        f.write(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}\n")
        f.write(f"========================================\n\n")

        # 先计算 dense baseline (None 策略) 的平均上传 KB, 用作 ratio 分母
        # 优先取整轮平均, 没有 None 则回退到 last-round max
        dense_baseline_kb = 0.0
        if "None" in all_results:
            none_metrics = all_results["None"]["client_metrics"]
            none_kbs = [m.get("avg_upload_size_kb", 0.0) for m in none_metrics]
            if none_kbs:
                dense_baseline_kb = sum(none_kbs) / len(none_kbs)
        if dense_baseline_kb <= 0.0:
            # 回退: 取所有实验中最大的上传 KB 作为近似 dense
            all_last_kbs = []
            for n in all_results:
                m = all_results[n]["client_metrics"]
                if m:
                    all_last_kbs.append(m[-1].get("avg_upload_size_kb", 0.0))
            dense_baseline_kb = max(all_last_kbs) if all_last_kbs else 1.0

        f.write(f"Dense baseline (avg upload KB): {dense_baseline_kb:.2f}\n")
        f.write("Ratio = avg_upload_kb / dense_baseline_kb  (用于与 FedMef/FedRTS 的 Data Exchange ratio 跨论文对比)\n\n")

        # 表头 (新增 Avg Upload 和 Ratio 两列)
        headers = ["Experiment", "Strategy", "Threshold", "Final Acc", "Best Acc",
                   "Last Upload(KB)", "Avg Upload(KB)", "Ratio(x dense)"]
        header_str = (f"{headers[0]:<12} | {headers[1]:<15} | {headers[2]:<10} | "
                      f"{headers[3]:<10} | {headers[4]:<10} | {headers[5]:<15} | "
                      f"{headers[6]:<15} | {headers[7]:<15}")
        f.write(header_str + "\n")
        f.write("-" * len(header_str) + "\n")

        for exp in EXPERIMENTS_CONFIG:
            name = exp["name"]
            if name not in all_results:
                continue

            strategy_name = exp["strategy"]
            threshold = exp["threshold"]

            results = all_results[name]
            # 获取精度数据
            accuracies_data = results["global_accuracies"]
            accuracies = [acc for _, acc in accuracies_data]

            final_acc = accuracies[-1] if accuracies else 0.0
            best_acc = max(accuracies) if accuracies else 0.0

            # 通信开销
            metrics = results["client_metrics"]
            last_round_metric = metrics[-1] if metrics else {}
            last_upload = last_round_metric.get("avg_upload_size_kb", 0.0)

            all_kbs = [m.get("avg_upload_size_kb", 0.0) for m in metrics]
            avg_upload = sum(all_kbs) / len(all_kbs) if all_kbs else 0.0

            ratio = avg_upload / dense_baseline_kb if dense_baseline_kb > 0 else 0.0

            row_str = (f"{name:<12} | {strategy_name:<15} | {str(threshold):<10} | "
                       f"{final_acc:.4f}     | {best_acc:.4f}     | {last_upload:<15.2f} | "
                       f"{avg_upload:<15.2f} | {ratio:<15.4f}")
            f.write(row_str + "\n")

    print(f"Summary saved to {filename}")

save_summary_to_txt()

