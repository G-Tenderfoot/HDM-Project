import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import flwr as fl
import torch.nn as nn
from torch.optim import SGD
from collections import OrderedDict
import time
import copy

# 导入模型和剪枝工具
from models import CNN, MnistCNN, ResNet18, ResNet20 # 根据你的模型
from pruning_utils import get_parameters_as_vector, set_parameters_from_vector, apply_pruning, \
                          state_dict_to_sparse_representation_for_upload, \
                          sparse_representation_to_state_dict_from_download


class CustomSubset(Dataset):
    """一个包装器，用于从原始数据集中选择指定索引的子集"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def load_datasets(dataset_name: str, num_clients: int, iid: bool = True, alpha: float = 0.1):
    """
    加载并根据IID或Non-IID策略将数据集划分到指定数量的客户端。
    增加了强制分配逻辑，防止 Alpha=0.1 时出现空客户端导致的死循环。
    """
    # 1. 加载原始数据集
    if dataset_name == "MNIST":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
        testset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset. Choose 'MNIST', 'CIFAR10' or 'CIFAR100'.")

    # 2. 数据划分逻辑
    client_data_indices = [[] for _ in range(num_clients)]

    if iid:
        # IID 划分 (随机打散)
        total_len = len(trainset)
        indices = list(range(total_len))
        np.random.shuffle(indices)

        partition_size = total_len // num_clients
        for i in range(num_clients):
            client_data_indices[i] = indices[i * partition_size: (i + 1) * partition_size]
        if total_len % num_clients != 0:
            client_data_indices[num_clients - 1].extend(indices[num_clients * partition_size:])
    else:
        # Non-IID 划分 (Dirichlet)
        labels = np.array(trainset.targets) if hasattr(trainset, 'targets') else np.array(trainset.labels)

        # 记录每个类别的索引
        class_indices = [np.where(labels == k)[0] for k in range(num_classes)]

        print(f"Generating Non-IID distribution (Alpha={alpha})...")

        # === 改进的核心逻辑 ===
        # 不再使用 while 循环死等，而是先生成，再补救

        # 1. 生成 Dirichlet 概率矩阵 (num_classes x num_clients)
        # 每一行代表一个类别在各个客户端的分布比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)

        # 2. 根据比例分配索引
        for k in range(num_classes):
            idx_k = class_indices[k]
            np.random.shuffle(idx_k)

            # 计算该类别分给每个客户端的数量
            # (最后可能会由舍入误差剩下几个样本，直接给最后一个客户端)
            proportions_k = np.array([p * (len(idx_k) / num_clients) for p in proportions[k]])

            # 重新归一化 proportions，以确保数量总和正确
            proportions_k = (proportions_k / proportions_k.sum()) * len(idx_k)
            proportions_k = proportions_k.astype(int)

            # 处理剩余的样本 (由于astype(int)可能会少分)
            current_idx = 0
            for i in range(num_clients):
                start = current_idx
                end = current_idx + proportions_k[i]
                client_data_indices[i].extend(idx_k[start:end].tolist())
                current_idx = end

            # 如果还有剩下的，随机分给某人
            if current_idx < len(idx_k):
                client_data_indices[np.random.randint(0, num_clients)].extend(idx_k[current_idx:].tolist())

        # 3. 【强制补救】检查并填充空客户端
        # 如果有客户端是空的，从数据最多的客户端那里“抢”一点数据过来
        min_require_samples = 10  # 保证每个客户端至少有10条数据

        for i in range(num_clients):
            if len(client_data_indices[i]) < min_require_samples:
                # 缺多少？
                needed = min_require_samples - len(client_data_indices[i])

                # 从谁那抢？(找数据量最多的客户端)
                # 注意：不要一直抢同一个富人，每次都重新找
                for _ in range(needed):
                    wealthiest_client = np.argmax([len(c) for c in client_data_indices])
                    if len(client_data_indices[wealthiest_client]) > min_require_samples:
                        # 抢一个样本
                        stolen_sample = client_data_indices[wealthiest_client].pop()
                        client_data_indices[i].append(stolen_sample)

        print(
            f"Non-IID distribution generated. Min samples: {min([len(c) for c in client_data_indices])}, Max samples: {max([len(c) for c in client_data_indices])}")

    client_datasets = [CustomSubset(trainset, indices) for indices in client_data_indices]
    return client_datasets, testset


class FlowerClient(fl.client.NumPyClient):
    """
    Flower联邦学习客户端的实现。
    负责本地训练、剪枝和与服务器的参数交换。
    """
    
    def __init__(self, cid: str, net: nn.Module, trainloader: DataLoader, valloader: DataLoader, 
                 device: torch.device):
        self.cid = cid
        self.net = net
        # 现在直接接收自己的加载器，而不是一个列表
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        
        self._initial_global_weights_vector = None
        
        self.model_state_dict_keys = list(self.net.state_dict().keys())

        self.static_config = {}

    def get_parameters(self, config: dict):
        """
        获取模型参数，按需应用剪枝，并以可序列化的列表格式返回。
        此方法在服务器请求初始参数时被调用。
        """
        # 优先读取 config，没有则读取 static_config
        pruning_strategy = config.get("pruning_strategy") or self.static_config.get("pruning_strategy", "none")
        threshold_param = config.get("threshold_param") or self.static_config.get("threshold_param", 0.0)
        delta_mad_scale = config.get("delta_mad_scale") or self.static_config.get("delta_mad_scale", 2.0)

        # === 修复点：Round 0 初始化保护 ===
        # 如果是第一次获取参数（Round 0），或者还没有初始权重向量，
        # 我们无法计算 Delta，因此强制将策略临时设为 "none"（不剪枝）。
        if self._initial_global_weights_vector is None:
            # print(f"Client {self.cid}: Initial weights None (Round 0?), skipping pruning.")
            actual_strategy = "none"
        else:
            actual_strategy = pruning_strategy

        pruned_state_dict = apply_pruning(
            self.net,
            pruning_strategy=actual_strategy,  # <--- 这里使用安全策略
            initial_weights_vector=self._initial_global_weights_vector,
            threshold_param=threshold_param,
            delta_mad_scale=delta_mad_scale
        )

        params_list, _ = state_dict_to_sparse_representation_for_upload(pruned_state_dict)

        return params_list

    def set_parameters(self, parameters: list):
        """
        从服务器接收参数并加载到模型中。
        """
        temp_model_keys = self.model_state_dict_keys

        # === 修改点：智能判断参数格式 ===
        if len(parameters) > 0 and parameters[0].size == 1:
            # 稀疏格式
            state_dict = sparse_representation_to_state_dict_from_download(
                parameters,
                temp_model_keys,
                self.device
            )
        else:
            # 密集格式
            state_dict = OrderedDict(
                {key: torch.from_numpy(param).to(self.device) for key, param in zip(temp_model_keys, parameters)}
            )

        self.net.load_state_dict(state_dict, strict=True)
        self._initial_global_weights_vector = get_parameters_as_vector(self.net)


    def fit(self, parameters: list, config: dict):
        """
        客户端在本地数据上训练模型。
        """
        self.set_parameters(parameters)
        
        epochs = config.get("epochs", 2)
        learning_rate = config.get("learning_rate", 0.01)
        client_momentum = config.get("client_momentum", 0.9)
        server_round = config.get("server_round", 0)

        # 处理没有数据的客户端
        if self.trainloader is None:
            print(f"Client {self.cid} has no data to train on. Skipping fit phase.")
            # 直接在这里处理参数，而不是调用 get_parameters
            pruned_state_dict = apply_pruning(self.net, **config)
            current_params, upload_size = state_dict_to_sparse_representation_for_upload(pruned_state_dict)
            return current_params, 0, {"fit_duration": 0.0, "upload_size_bytes": upload_size}

        # 本地训练循环
        optimizer = SGD(
            self.net.parameters(),
            lr=learning_rate,
            momentum=client_momentum,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()
        
        self.net.train()
        start_time = time.time()
        for epoch in range(epochs):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        fit_duration = time.time() - start_time
        print(f"Client {self.cid} - Round {server_round} Fit duration: {fit_duration:.2f}s")

        # 训练结束后，不要调用 get_parameters。
        # 直接在这里完成剪枝和打包，以同时获得参数和上传大小。
        pruned_state_dict = apply_pruning(
            self.net,
            pruning_strategy=config.get("pruning_strategy", "none"),
            initial_weights_vector=self._initial_global_weights_vector,
            threshold_param=config.get("threshold_param", 0.9),
            delta_mad_scale=config.get("delta_mad_scale", 2.0)
        )
        updated_params, upload_size = state_dict_to_sparse_representation_for_upload(pruned_state_dict)
        
        # 将上传大小放入metrics字典
        metrics = {"fit_duration": fit_duration, "upload_size_bytes": upload_size}
        
        num_examples = len(self.trainloader.dataset)
        return updated_params, num_examples, metrics
    
    def evaluate(self, parameters: list, config: dict):
        """
        客户端在本地验证集上评估模型。
        """
        self.set_parameters(parameters) # 加载服务器下发的全局模型
        self.net.eval() # 设置模型为评估模式
        
        loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        
        server_round = config.get("server_round", "?") # 使用 '?' 作为默认值，方便调试
        
        with torch.no_grad(): # 在评估时不计算梯度
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss /= len(self.valloader.dataset)
        accuracy = correct / len(self.valloader.dataset)
        print(f"Client {self.cid} - Round {server_round} Eval loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        # 返回损失、样本数和准确率
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
