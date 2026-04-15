import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from models import CNN, MnistCNN, ResNet18, ResNet20
from pruning_utils import sparse_representation_to_state_dict_from_download, \
                          apply_pruning, get_parameters_as_vector, state_dict_to_sparse_representation_for_upload
from client import load_datasets

class PruningFedAvg(fl.server.strategy.FedAvg):
    def __init__(
            self,
            dataset_name: str,
            model_name: str = "CNN", # Added model_name
            pruning_strategy: str = "none",
            threshold_param: float = 0.0,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn = None,
            on_fit_config_fn = None,
            on_evaluate_config_fn = None,
            initial_parameters = None,
            strategy_config: dict = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
        )
        self.dataset_name = dataset_name
        self.model_name = model_name # Store model_name
        self.pruning_strategy = pruning_strategy
        self.threshold_param = threshold_param
        self.strategy_config = strategy_config or {}

        self.global_model_vector = None
        self.metrics_aggregated_per_round = []

    def _get_temp_model(self):
        """Helper to create a temporary model instance for shape information."""
        # Determine num_classes based on dataset
        if self.dataset_name == "CIFAR100":
            num_classes = 100
        else:
            num_classes = 10

        if self.model_name == "ResNet18":
            return ResNet18(num_classes=num_classes)
        elif self.model_name == "ResNet20":
            return ResNet20(num_classes=num_classes)
        elif self.dataset_name == "MNIST":
            return MnistCNN(num_classes=num_classes)
        else:
            return CNN(num_classes=num_classes)

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters,
                      client_manager: fl.server.client_manager.ClientManager):
        params_list = fl.common.parameters_to_ndarrays(parameters)
        temp_model = self._get_temp_model()
        temp_model_keys = list(temp_model.state_dict().keys())

        # === 修改点 1：智能判断参数格式 ===
        # 检查第一个参数的 size。如果是 1，说明是 Flag (稀疏格式)；否则是权重矩阵 (密集格式)。
        if len(params_list) > 0 and params_list[0].size == 1:
            # 稀疏格式 -> 解码
            state_dict = sparse_representation_to_state_dict_from_download(
                params_list, temp_model_keys, torch.device("cpu")
            )
        else:
            # 密集格式 -> 直接加载
            state_dict = OrderedDict(
                {key: torch.from_numpy(param) for key, param in zip(temp_model_keys, params_list)}
            )

        temp_model.load_state_dict(state_dict)
        global_weights_vector = get_parameters_as_vector(temp_model)

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        config["pruning_strategy"] = self.pruning_strategy
        config["threshold_param"] = self.threshold_param
        config["server_round"] = server_round

        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters,
                           client_manager: fl.server.client_manager.ClientManager):
        eval_configurations = super().configure_evaluate(server_round, parameters, client_manager)
        for client_proxy, eval_ins in eval_configurations:
            eval_ins.config["server_round"] = server_round
        return eval_configurations

    def aggregate_fit(self, server_round: int,
                      results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]):
        fit_durations = [r.metrics["fit_duration"] for _, r in results if "fit_duration" in r.metrics]
        upload_sizes_kb = [r.metrics["upload_size_bytes"] / 1024 for _, r in results if
                           "upload_size_bytes" in r.metrics]

        avg_fit_duration = np.mean(fit_durations) if fit_durations else 0
        avg_upload_size_kb = np.mean(upload_sizes_kb) if upload_sizes_kb else 0

        decoded_results_for_agg = []
        temp_model = self._get_temp_model()
        temp_model_keys = list(temp_model.state_dict().keys())

        for client_proxy, fit_res in results:
            params_list = fl.common.parameters_to_ndarrays(fit_res.parameters)

            # 客户端上传的肯定是稀疏格式，所以这里直接解码没问题
            # 但为了保险，也可以加上判断
            if len(params_list) > 0 and params_list[0].size == 1:
                state_dict = sparse_representation_to_state_dict_from_download(
                    params_list, temp_model_keys, torch.device("cpu")
                )
            else:
                state_dict = OrderedDict(
                    {key: torch.from_numpy(param) for key, param in zip(temp_model_keys, params_list)}
                )

            dense_ndarrays = [val.cpu().numpy() for val in state_dict.values()]

            new_fit_res = fl.common.FitRes(
                status=fit_res.status,
                parameters=fl.common.ndarrays_to_parameters(dense_ndarrays),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            decoded_results_for_agg.append((client_proxy, new_fit_res))

        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, decoded_results_for_agg,
                                                                          failures)

        self.metrics_aggregated_per_round.append({
            "round": server_round,
            "avg_fit_duration": avg_fit_duration,
            "avg_upload_size_kb": avg_upload_size_kb,
        })
        print(
            f"Server - Round {server_round} Aggregated - Avg Fit: {avg_fit_duration:.2f}s, Avg Upload: {avg_upload_size_kb:.2f}KB")
        return aggregated_parameters, metrics_aggregated


def evaluate_global_model(server_round: int, parameters: list, config: dict):
    dataset_name = config.get("dataset_name", "CIFAR10")
    model_name = config.get("model_name", "CNN")
    batch_size = config.get("batch_size", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "CIFAR100":
        num_classes = 100
    else:
        num_classes = 10

    if model_name == "ResNet18":
        net = ResNet18(num_classes=num_classes)
    elif model_name == "ResNet20":
        net = ResNet20(num_classes=num_classes)
    elif dataset_name == "MNIST":
        net = MnistCNN(num_classes=num_classes)
    else:
        net = CNN(num_classes=num_classes)

    net.to(device)

    ndarrays_params = parameters
    model_state_dict_keys = list(net.state_dict().keys())

    # === 修改点 2：智能判断参数格式 (这里是报错的源头) ===
    # 判断是否为稀疏格式 (Flag size == 1)
    if len(ndarrays_params) > 0 and ndarrays_params[0].size == 1:
        state_dict = sparse_representation_to_state_dict_from_download(
            ndarrays_params, model_state_dict_keys, device
        )
    else:
        # 聚合后的密集格式
        state_dict = OrderedDict(
            {key: torch.from_numpy(param).to(device) for key, param in zip(model_state_dict_keys, ndarrays_params)}
        )

    net.load_state_dict(state_dict)
    net.eval()
    loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    # 使用较小的测试集子集加速评估，或者使用全量测试集
    _, testset = load_datasets(dataset_name, num_clients=100, iid=True)
    testloader = DataLoader(testset, batch_size=batch_size)

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(testset)
    accuracy = correct / len(testset)
    print(f"Server - Global Eval Round {server_round} - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
    return loss, {"accuracy": accuracy}