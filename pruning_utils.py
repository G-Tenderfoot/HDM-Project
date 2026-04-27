import flwr
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import copy
from scipy.sparse import csr_matrix, issparse

def get_parameters_as_vector(model: nn.Module) -> np.ndarray:
    """
    将模型的所有参数（权重和偏置）展平为一个NumPy数组。
    """
    param_list = []
    for param in model.parameters():
        param_list.append(param.data.cpu().numpy().flatten())
    return np.concatenate(param_list)

def set_parameters_from_vector(model: nn.Module, param_vector: np.ndarray):
    """
    从一个扁平化的NumPy数组中恢复模型参数，并设置回模型。
    """
    pointer = 0
    for param in model.parameters():
        num_elements = param.numel()
        # 确保数据类型匹配，并且移动到正确的设备
        param.data = torch.from_numpy(param_vector[pointer:pointer + num_elements].reshape(param.shape)).to(param.device).float()
        pointer += num_elements

def calculate_mad(weights: np.ndarray) -> float:
    """
    计算给定权重数组的平均绝对偏差 (MAD)。
    """
    if len(weights) == 0:
        return 0.0
    return np.mean(np.abs(weights - np.mean(weights)))


def apply_pruning(model: nn.Module, pruning_strategy: str, initial_weights_vector: np.ndarray = None,
                  threshold_param: float = 0.9, delta_mad_scale: float = 2.0):
    """
    根据不同的策略对模型参数进行剪枝。
    """
    pruned_model = copy.deepcopy(model)
    current_flat_weights = get_parameters_as_vector(model)

    if pruning_strategy == "none":
        return model.state_dict()

    pruned_flat_weights = current_flat_weights.copy()

    if pruning_strategy == "magnitude":
        # === 修改点1：修正百分位计算 ===
        # threshold_param=0.95 意味着剪掉 95%，保留最大的 5%
        flat_abs_weights = np.abs(current_flat_weights)
        threshold = np.percentile(flat_abs_weights, threshold_param * 100)
        pruned_flat_weights[flat_abs_weights < threshold] = 0

    elif pruning_strategy == "delta_prun_mad":
        if initial_weights_vector is None:
            raise ValueError("initial_weights_vector must be provided for delta_prun_mad pruning.")

        delta_weights = np.abs(current_flat_weights - initial_weights_vector)
        mad_threshold = calculate_mad(delta_weights) * threshold_param
        pruned_flat_weights[delta_weights < mad_threshold] = 0

    elif pruning_strategy == "update_prun_mad":
        mad_threshold = calculate_mad(current_flat_weights) * threshold_param
        pruned_flat_weights[np.abs(current_flat_weights) < mad_threshold] = 0

    elif pruning_strategy == "hdm_prun":
        if initial_weights_vector is None:
            raise ValueError("initial_weights_vector must be provided for hdm_prun.")

        # # 1. Delta Mask (找出正在积极学习的参数)
        # delta_weights = np.abs(current_flat_weights - initial_weights_vector)
        # # 调低 MAD 倍数，因为取交集已经很严格了，设为 1.0 或 1.5 即可
        # delta_mad_threshold = calculate_mad(delta_weights) * 0.4
        # change_mask = delta_weights >= delta_mad_threshold
        #
        # # 2. Magnitude Mask (找出包含核心特征的参数)
        # magnitude_threshold = np.percentile(np.abs(current_flat_weights), threshold_param * 100)
        # significance_mask = np.abs(current_flat_weights) >= magnitude_threshold
        #
        # # 3. 【核心修改】取交集 (Logical AND)
        # # 只有既重要、又在积极学习的参数才会被保留和上传
        # final_mask = np.logical_and(change_mask, significance_mask)
        # pruned_flat_weights[~final_mask] = 0

        # 1. Delta Mask
        delta_weights = np.abs(current_flat_weights - initial_weights_vector)
        delta_mad_threshold = calculate_mad(delta_weights) * delta_mad_scale
        change_mask = np.abs(delta_weights) >= delta_mad_threshold

        # 2. Magnitude Mask
        # 使用 threshold_param (如 0.95) 控制保留比例
        magnitude_threshold = np.percentile(np.abs(current_flat_weights), threshold_param * 100)
        significance_mask = np.abs(current_flat_weights) >= magnitude_threshold

        # 3. 取并集
        final_mask = np.logical_or(change_mask, significance_mask)
        pruned_flat_weights[~final_mask] = 0

    else:
        raise ValueError(f"Unknown pruning strategy: {pruning_strategy}")

    set_parameters_from_vector(pruned_model, pruned_flat_weights)
    return pruned_model.state_dict()


def state_dict_to_sparse_representation_for_upload(state_dict: OrderedDict) -> tuple[list, int]:
    """
    将模型state_dict转换为适合Flower NumPyClient上传的格式，并估算通信量。
    对于高维张量（如卷积核），先将其重塑为2D再进行稀疏化。
    
    返回:
    - list: 包含所有参数（可能稀疏化后分解为多个NumPy数组）的列表。
    - int: 估算的通信量（字节）。
    """
    params_for_upload = []
    total_data_size = 0

    for key, value in state_dict.items():
        np_value = value.cpu().numpy()
        original_shape = np_value.shape

        # 只对权重或大型偏置进行稀疏化
        is_prunable = 'weight' in key or ('bias' in key and np_value.ndim > 0 and np_value.size > 100)

        if is_prunable:
            np_value_2d = np_value.reshape(original_shape[0], -1) if np_value.ndim > 1 else np_value.reshape(1, -1)
            sparse_matrix = csr_matrix(np_value_2d)
            
            # 只有当稀疏表示更节省空间时才使用它
            if sparse_matrix.nnz < np_value.size / 2: 
                params_for_upload.extend([
                    np.array([1], dtype=np.int8), 
                    sparse_matrix.data.astype(np.float32), 
                    sparse_matrix.indices.astype(np.int32), 
                    sparse_matrix.indptr.astype(np.int32), 
                    np.array(sparse_matrix.shape, dtype=np.int32),
                    np.array(original_shape, dtype=np.int32)
                ])
                total_data_size += (sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + 
                                    sparse_matrix.indptr.nbytes + 12) # 估算形状数组的大小
            else:
                params_for_upload.extend([np.array([0], dtype=np.int8), np_value])
                total_data_size += np_value.nbytes
        else:
            params_for_upload.extend([np.array([0], dtype=np.int8), np_value])
            total_data_size += np_value.nbytes
            
    return params_for_upload, total_data_size


def sparse_representation_to_state_dict_from_download(parameters: list, original_model_state_dict_keys: list, device: torch.device) -> OrderedDict:
    """
    将从服务器下载的参数（NumPy数组列表，可能包含稀疏表示的分解）
    恢复为模型state_dict。
    
    参数:
    - parameters (list): 从服务器接收的NumPy数组列表。
    - original_model_state_dict_keys (list): 原始模型state_dict的所有键，用于重建顺序。
    - device (torch.device): 模型将加载到的设备。
    
    返回:
    - OrderedDict: 恢复的state_dict。
    """
    state_dict = OrderedDict()
    param_idx = 0
    current_key_idx = 0

    while param_idx < len(parameters):
        if current_key_idx >= len(original_model_state_dict_keys):
            break # 避免索引越界
        key = original_model_state_dict_keys[current_key_idx]
        is_sparse_flag = parameters[param_idx].item()
        param_idx += 1

        if is_sparse_flag == 1: # 稀疏表示
            data = parameters[param_idx]
            indices = parameters[param_idx + 1]
            indptr = parameters[param_idx + 2]
            shape_2d = tuple(parameters[param_idx + 3])
            original_shape = tuple(parameters[param_idx + 4]) # *** 接收原始形状 ***
            param_idx += 5
            
            sparse_matrix = csr_matrix((data, indices, indptr), shape=shape_2d)
            np_value = sparse_matrix.toarray()
            # *** 使用原始形状进行重塑 ***
            np_value = np_value.reshape(original_shape)

        else: # 非稀疏表示
            np_value = parameters[param_idx]
            param_idx += 1
        
        state_dict[key] = torch.from_numpy(np_value).to(device).float()
        current_key_idx += 1
        
    return state_dict