import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict
import pickle
import gc
import ray
import matplotlib.pyplot as plt

# 导入自定义模块
from models import CNN, MnistCNN, ResNet18, ResNet20
from client import FlowerClient, load_datasets
from server import PruningFedAvg, evaluate_global_model
from pruning_utils import get_parameters_as_vector, set_parameters_from_vector

# 全局配置
NUM_CLIENTS = 100
NUM_ROUNDS = 200
FRACTION_FIT = 0.2    # 每轮训练的客户端比例
FRACTION_EVALUATE = 0.2 # 每轮评估的客户端比例
BATCH_SIZE = 128
CLIENT_EPOCHS = 2
LEARNING_RATE = 0.005

DATASET_NAME = "CIFAR10"  # 'MNIST', 'CIFAR10', 'CIFAR100'
MODEL_NAME = "ResNet20" # 'CNN', 'ResNet18', 'ResNet20'
IID = False            # 是否IID
ALPHA = 1.0           # Non-IIDDirichlet参数

PRUNING_STRATEGY = "hdm"  # 'l1', 'random', 'hdm', 'mp', 'mag'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# 2. 定义四个实验组 (对照实验)
# ==========================================
EXPERIMENTS_CONFIG = [
    # {
    #     "name": "None",
    #     "strategy": "none",
    #     "threshold": 0.0,
    #     "color": "tab:blue"
    # },
    {
        "name": "Magnitude",
        "strategy": "magnitude",
        "threshold": 0.70,
        "color": "tab:orange"
    }
    # {
    #     "name": "Delta",
    #     "strategy": "delta_prun_mad",
    #     "threshold": 0.6,  # 2倍MAD，确保剪得够狠，后期才会崩
    #     "color": "tab:green"
    # },
    # {
    #     "name": "HDM",
    #     "strategy": "hdm_prun",
    #     "threshold": 0.75,
    #     "color": "tab:purple"
    # }
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
for exp in EXPERIMENTS_CONFIG:
    exp_name = exp["name"]
    strategy_name = exp["strategy"]
    threshold_val = exp["threshold"]

    print(f"\n>>> Running Experiment: {exp_name} (Strategy: {strategy_name}, Threshold: {threshold_val})")


    # 定义Client工厂 (闭包传参)
    def make_client_fn(client_datasets_list, device, s_name, t_val):
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
            client.static_config = {"pruning_strategy": s_name, "threshold_param": t_val}
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
            "dataset_name": DATASET_NAME, "server_round": server_round,
            "pruning_strategy": strategy_name, "threshold_param": threshold_val
        },
    )

    # 启动模拟
    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(client_datasets, DEVICE, strategy_name, threshold_val),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.05},  # 4090D 资源配置
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )

    # 保存结果
    all_results[exp_name] = {
        "global_accuracies": history.metrics_centralized.get("accuracy", []),
        "client_metrics": strategy.metrics_aggregated_per_round
    }

    ray.shutdown()
    torch.cuda.empty_cache()
    gc.collect()


# ==========================================
# 4. 自动绘图函数 (一鱼三吃)
# ==========================================
def generate_plot(title, filename, strategies_to_show, max_round):
    print(f"Generating plot: {filename}...")
    plt.figure(figsize=(20, 6))

    # 子图1: Accuracy
    plt.subplot(1, 3, 1)
    for exp_conf in EXPERIMENTS_CONFIG:
        name = exp_conf["name"]
        if name not in strategies_to_show: continue

        data = all_results[name]
        rounds, accs = zip(*data["global_accuracies"])
        # 截取前 max_round 轮
        rounds = [r for r in rounds if r <= max_round]
        accs = accs[:len(rounds)]
        plt.plot(rounds, accs, label=name, color=exp_conf["color"], linewidth=2)

    plt.title("Global Accuracy")
    plt.xlabel("Rounds");
    plt.ylabel("Accuracy");
    plt.grid(True);
    plt.legend()

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

    plt.title("Avg Upload Size (KB)")
    plt.xlabel("Rounds");
    plt.grid(True);
    plt.legend()

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

    plt.title("Avg Fit Duration (s)")
    plt.xlabel("Rounds");
    plt.grid(True);
    plt.legend()

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

# 实验3：验证 HDM 优势 (前25轮，全对比)
generate_plot(
    title="Exp 3: HDM Performance (Round 0-200)",
    filename="exp3_hdm_final.png",
    strategies_to_show=["None", "Magnitude", "Delta", "HDM"],
    max_round=200
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

        # 表头
        headers = ["Experiment", "Strategy", "Threshold", "Final Acc", "Best Acc", "Last Upload(KB)"]
        # 格式化字符串
        header_str = f"{headers[0]:<12} | {headers[1]:<15} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<10} | {headers[5]:<15}"
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
            # accuracies_data 格式是 [(round, acc), ...]
            accuracies = [acc for _, acc in accuracies_data]

            final_acc = accuracies[-1] if accuracies else 0.0
            best_acc = max(accuracies) if accuracies else 0.0

            # 获取通信开销数据
            metrics = results["client_metrics"]
            last_round_metric = metrics[-1] if metrics else {}
            avg_upload = last_round_metric.get("avg_upload_size_kb", 0.0)

            row_str = f"{name:<12} | {strategy_name:<15} | {str(threshold):<10} | {final_acc:.4f}     | {best_acc:.4f}     | {avg_upload:.2f}"
            f.write(row_str + "\n")

    print(f"Summary saved to {filename}")

save_summary_to_txt()

