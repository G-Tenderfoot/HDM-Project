import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from scipy.sparse import csr_matrix

from models import CNN, MnistCNN, ResNet18, ResNet20
from pruning_utils import (
    apply_pruning,
    get_parameters_as_vector,
    sparse_representation_to_state_dict_from_download,
    state_dict_to_sparse_representation_for_upload,
)
from client import load_datasets
import recorder


class PruningFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        dataset_name: str,
        model_name: str = "CNN",
        pruning_strategy: str = "none",
        threshold_param: float = 0.0,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        initial_parameters=None,
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
        self.model_name = model_name
        self.pruning_strategy = pruning_strategy
        self.threshold_param = threshold_param
        self.strategy_config = strategy_config or {}

        self.metrics_aggregated_per_round = []
        self.server_momentum_beta = self.strategy_config.get("server_momentum_beta", 0.0)
        self.delta_mad_scale = self.strategy_config.get("delta_mad_scale", 2.0)
        self.sparse_downlink = self.strategy_config.get("sparse_downlink", True)
        self.last_downlink_size_kb = 0.0
        self.momentum_buffer = None
        self.current_global_weights = (
            [np.array(arr, copy=True) for arr in fl.common.parameters_to_ndarrays(initial_parameters)]
            if initial_parameters is not None else None
        )

    def _get_temp_model(self):
        if self.dataset_name == "CIFAR100":
            num_classes = 100
        else:
            num_classes = 10

        if self.model_name == "ResNet18":
            return ResNet18(num_classes=num_classes)
        if self.model_name == "ResNet20":
            return ResNet20(num_classes=num_classes)
        if self.dataset_name == "MNIST":
            return MnistCNN(num_classes=num_classes)
        return CNN(num_classes=num_classes)

    def _get_model_metadata(self):
        temp_model = self._get_temp_model()
        reference_state_dict = temp_model.state_dict()
        state_keys = list(reference_state_dict.keys())
        trainable_keys = set(dict(temp_model.named_parameters()).keys())
        return reference_state_dict, state_keys, trainable_keys

    def _ndarrays_to_reference_state_dict(
        self,
        ndarrays: list[np.ndarray],
        reference_state_dict: OrderedDict,
        device: torch.device = torch.device("cpu"),
    ) -> OrderedDict:
        state_dict = OrderedDict()
        for (key, ref_tensor), array in zip(reference_state_dict.items(), ndarrays):
            tensor = torch.from_numpy(array).to(device=device, dtype=ref_tensor.dtype)
            state_dict[key] = tensor
        return state_dict

    def _state_dict_to_ndarrays(self, state_dict: OrderedDict) -> list[np.ndarray]:
        return [value.cpu().numpy() for value in state_dict.values()]

    def _dense_size_bytes(self, ndarrays: list[np.ndarray]) -> int:
        return int(sum(np.asarray(array).nbytes for array in ndarrays))

    def _build_model_from_ndarrays(
        self,
        ndarrays: list[np.ndarray],
        reference_state_dict: OrderedDict,
    ) -> nn.Module:
        model = self._get_temp_model()
        state_dict = self._ndarrays_to_reference_state_dict(ndarrays, reference_state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def _apply_server_side_pruning(
        self,
        ndarrays: list[np.ndarray],
        base_weights: list[np.ndarray] | None,
        reference_state_dict: OrderedDict,
    ) -> list[np.ndarray]:
        if (not self.sparse_downlink) or self.pruning_strategy == "none":
            return ndarrays

        model = self._build_model_from_ndarrays(ndarrays, reference_state_dict)
        initial_weights_vector = None

        if self.pruning_strategy in ("hdm_prun", "delta_prun_mad"):
            if base_weights is None:
                return ndarrays
            base_model = self._build_model_from_ndarrays(base_weights, reference_state_dict)
            initial_weights_vector = get_parameters_as_vector(base_model)

        pruned_state_dict = apply_pruning(
            model,
            pruning_strategy=self.pruning_strategy,
            initial_weights_vector=initial_weights_vector,
            threshold_param=self.threshold_param,
            delta_mad_scale=self.delta_mad_scale,
        )
        return self._state_dict_to_ndarrays(pruned_state_dict)

    def _apply_initial_downlink_pruning(
        self,
        ndarrays: list[np.ndarray],
        reference_state_dict: OrderedDict,
    ) -> list[np.ndarray]:
        if (not self.sparse_downlink) or self.pruning_strategy == "none":
            return ndarrays

        initial_strategy = self.pruning_strategy
        if initial_strategy in ("hdm_prun", "delta_prun_mad"):
            initial_strategy = "magnitude"

        model = self._build_model_from_ndarrays(ndarrays, reference_state_dict)
        pruned_state_dict = apply_pruning(
            model,
            pruning_strategy=initial_strategy,
            initial_weights_vector=None,
            threshold_param=self.threshold_param,
            delta_mad_scale=self.delta_mad_scale,
        )
        return self._state_dict_to_ndarrays(pruned_state_dict)

    def _make_downlink_parameters(
        self,
        ndarrays: list[np.ndarray],
        reference_state_dict: OrderedDict,
    ) -> tuple[fl.common.Parameters, float]:
        if (not self.sparse_downlink) or self.pruning_strategy == "none":
            return fl.common.ndarrays_to_parameters(ndarrays), self._dense_size_bytes(ndarrays) / 1024

        state_dict = self._ndarrays_to_reference_state_dict(ndarrays, reference_state_dict)
        sparse_params, downlink_size_bytes = state_dict_to_sparse_representation_for_upload(state_dict)
        return fl.common.ndarrays_to_parameters(sparse_params), downlink_size_bytes / 1024

    def _decode_uploaded_parameters(
        self,
        parameters: list[np.ndarray],
        reference_state_dict: OrderedDict,
        base_state_dict: OrderedDict | None = None,
    ) -> OrderedDict:
        """Decode client uploads. Missing sparse coordinates remain zero unless a base state is provided."""
        state_dict = OrderedDict()
        state_keys = list(reference_state_dict.keys())
        param_idx = 0
        key_idx = 0

        while param_idx < len(parameters) and key_idx < len(state_keys):
            key = state_keys[key_idx]
            ref_tensor = reference_state_dict[key]
            is_sparse_flag = int(parameters[param_idx].item())
            param_idx += 1

            if is_sparse_flag == 1:
                data = parameters[param_idx]
                indices = parameters[param_idx + 1]
                indptr = parameters[param_idx + 2]
                shape_2d = tuple(parameters[param_idx + 3])
                original_shape = tuple(parameters[param_idx + 4])
                param_idx += 5

                sparse_matrix = csr_matrix((data, indices, indptr), shape=shape_2d)
                sparse_value = sparse_matrix.toarray().reshape(original_shape)

                if base_state_dict is not None and key in base_state_dict:
                    base_value = base_state_dict[key].cpu().numpy()
                    merged_value = np.array(base_value, copy=True)
                    nonzero_mask = sparse_value != 0
                    merged_value[nonzero_mask] = sparse_value[nonzero_mask]
                    np_value = merged_value
                else:
                    np_value = sparse_value
            else:
                np_value = parameters[param_idx]
                param_idx += 1

            state_dict[key] = torch.from_numpy(np_value).to(dtype=ref_tensor.dtype)
            key_idx += 1

        return state_dict

    def _sanitize_ndarrays(
        self,
        ndarrays: list[np.ndarray],
        state_keys: list[str],
        base_weights: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        safe_ndarrays = []

        for idx, (key, array) in enumerate(zip(state_keys, ndarrays)):
            safe_array = np.array(array, copy=True)
            base_array = None if base_weights is None else base_weights[idx]

            if np.issubdtype(safe_array.dtype, np.floating):
                if not np.all(np.isfinite(safe_array)):
                    if base_array is not None:
                        finite_mask = np.isfinite(safe_array)
                        safe_array = np.where(finite_mask, safe_array, base_array)
                    else:
                        safe_array = np.nan_to_num(safe_array, nan=0.0, posinf=0.0, neginf=0.0)

                if key.endswith("running_var"):
                    safe_array = np.maximum(safe_array, 1e-6)

            safe_ndarrays.append(safe_array)

        return safe_ndarrays

    def _apply_server_momentum(
        self,
        new_weights: list[np.ndarray],
        base_weights: list[np.ndarray] | None,
        state_keys: list[str],
        trainable_keys: set[str],
    ) -> list[np.ndarray]:
        if self.server_momentum_beta <= 0 or base_weights is None:
            return new_weights

        if self.momentum_buffer is None:
            self.momentum_buffer = []
            for key, array in zip(state_keys, new_weights):
                if key in trainable_keys and np.issubdtype(np.asarray(array).dtype, np.floating):
                    self.momentum_buffer.append(np.zeros_like(array, dtype=np.float32))
                else:
                    self.momentum_buffer.append(None)

        updated_weights = []
        for idx, (key, new_array, base_array) in enumerate(zip(state_keys, new_weights, base_weights)):
            if key in trainable_keys and np.issubdtype(np.asarray(new_array).dtype, np.floating):
                pseudo_gradient = new_array - base_array
                self.momentum_buffer[idx] = (
                    self.server_momentum_beta * self.momentum_buffer[idx]
                    + (1 - self.server_momentum_beta) * pseudo_gradient
                )
                updated_weights.append(base_array + self.momentum_buffer[idx])
            else:
                updated_weights.append(new_array)

        return updated_weights

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        params_list = fl.common.parameters_to_ndarrays(parameters)
        self.current_global_weights = [np.array(arr, copy=True) for arr in params_list]

        reference_state_dict, temp_model_keys, _ = self._get_model_metadata()

        if len(params_list) > 0 and params_list[0].size == 1:
            state_dict = sparse_representation_to_state_dict_from_download(
                params_list, temp_model_keys, torch.device("cpu")
            )
        else:
            state_dict = self._ndarrays_to_reference_state_dict(params_list, reference_state_dict)

        temp_model = self._get_temp_model()
        temp_model.load_state_dict(state_dict, strict=True)
        broadcast_ndarrays = self._state_dict_to_ndarrays(state_dict)
        if server_round == 1:
            broadcast_ndarrays = self._apply_initial_downlink_pruning(
                broadcast_ndarrays,
                reference_state_dict,
            )
            self.current_global_weights = [np.array(arr, copy=True) for arr in broadcast_ndarrays]
        broadcast_parameters, downlink_size_kb = self._make_downlink_parameters(
            broadcast_ndarrays,
            reference_state_dict,
        )
        self.last_downlink_size_kb = downlink_size_kb

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        config["pruning_strategy"] = self.pruning_strategy
        config["threshold_param"] = self.threshold_param
        config["server_round"] = server_round

        fit_ins = fl.common.FitIns(broadcast_parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        eval_configurations = super().configure_evaluate(server_round, parameters, client_manager)
        for client_proxy, eval_ins in eval_configurations:
            eval_ins.config["server_round"] = server_round
        return eval_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ):
        fit_durations = [r.metrics["fit_duration"] for _, r in results if "fit_duration" in r.metrics]
        upload_sizes_kb = [
            r.metrics["upload_size_bytes"] / 1024 for _, r in results if "upload_size_bytes" in r.metrics
        ]

        avg_fit_duration = np.mean(fit_durations) if fit_durations else 0
        avg_upload_size_kb = np.mean(upload_sizes_kb) if upload_sizes_kb else 0
        avg_download_size_kb = self.last_downlink_size_kb
        avg_exchange_size_kb = avg_upload_size_kb + avg_download_size_kb

        reference_state_dict, temp_model_keys, trainable_keys = self._get_model_metadata()
        base_weights_for_round = (
            [np.array(arr, copy=True) for arr in self.current_global_weights]
            if self.current_global_weights is not None else None
        )

        decoded_results_for_agg = []
        for client_proxy, fit_res in results:
            params_list = fl.common.parameters_to_ndarrays(fit_res.parameters)
            decoded_state_dict = self._decode_uploaded_parameters(
                params_list,
                reference_state_dict,
                base_state_dict=None,
            )
            dense_ndarrays = [val.cpu().numpy() for val in decoded_state_dict.values()]

            new_fit_res = fl.common.FitRes(
                status=fit_res.status,
                parameters=fl.common.ndarrays_to_parameters(dense_ndarrays),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            decoded_results_for_agg.append((client_proxy, new_fit_res))

        aggregated_parameters, metrics_aggregated = super().aggregate_fit(
            server_round,
            decoded_results_for_agg,
            failures,
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            aggregated_ndarrays = self._sanitize_ndarrays(
                aggregated_ndarrays,
                temp_model_keys,
                base_weights=base_weights_for_round,
            )
            aggregated_ndarrays = self._apply_server_momentum(
                aggregated_ndarrays,
                base_weights_for_round,
                temp_model_keys,
                trainable_keys,
            )
            aggregated_ndarrays = self._sanitize_ndarrays(
                aggregated_ndarrays,
                temp_model_keys,
                base_weights=base_weights_for_round,
            )
            aggregated_ndarrays = self._apply_server_side_pruning(
                aggregated_ndarrays,
                base_weights_for_round,
                reference_state_dict,
            )
            aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_ndarrays)
            self.current_global_weights = [np.array(arr, copy=True) for arr in aggregated_ndarrays]

        self.metrics_aggregated_per_round.append(
            {
                "round": server_round,
                "avg_fit_duration": avg_fit_duration,
                "avg_upload_size_kb": avg_upload_size_kb,
                "avg_download_size_kb": avg_download_size_kb,
                "avg_exchange_size_kb": avg_exchange_size_kb,
            }
        )
        recorder.record_fit_metrics(
            server_round,
            avg_fit_duration,
            avg_upload_size_kb,
            avg_download_size_kb,
        )
        print(
            f"Server - Round {server_round} Aggregated - "
            f"Avg Fit: {avg_fit_duration:.2f}s, "
            f"Avg Upload: {avg_upload_size_kb:.2f}KB, "
            f"Avg Download: {avg_download_size_kb:.2f}KB"
        )
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
    reference_state_dict = net.state_dict()

    if len(ndarrays_params) > 0 and ndarrays_params[0].size == 1:
        state_dict = sparse_representation_to_state_dict_from_download(
            ndarrays_params, model_state_dict_keys, device
        )
    else:
        state_dict = OrderedDict()
        for key, ref_tensor, param in zip(model_state_dict_keys, reference_state_dict.values(), ndarrays_params):
            array = np.array(param, copy=True)
            if np.issubdtype(array.dtype, np.floating):
                array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                if key.endswith("running_var"):
                    array = np.maximum(array, 1e-6)
            state_dict[key] = torch.from_numpy(array).to(device=device, dtype=ref_tensor.dtype)

    net.load_state_dict(state_dict, strict=True)
    net.eval()
    loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()

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
    recorder.record_eval_metrics(server_round, loss, accuracy)
    return loss, {"accuracy": accuracy}
