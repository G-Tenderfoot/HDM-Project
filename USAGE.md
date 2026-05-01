# HDM 联邦学习实验 使用指南

本代码库用于在 CIFAR-10 + ResNet-18/20 上复现 HDM (Hybrid Delta-Magnitude) 剪枝策略, 并与 FedMef (CVPR'24) / FedRTS (NeurIPS'25) 的实验设置对齐做横向对比.

## 目录结构

```
code/
  main.py                 训练入口
  client.py               Flower 客户端 + 数据集加载 + Dirichlet 非 IID 划分
  server.py               PruningFedAvg 策略 + 全局评估函数
  pruning_utils.py        剪枝算法 + CSR 稀疏序列化
  models.py               CNN / MnistCNN / ResNet-18 / ResNet-20
  recorder.py             每轮 JSONL 记录器 (线程安全)
  plot_from_records.py    事后重绘图脚本 (不重跑训练)
  records/                训练产物 (JSONL + pickle)
```

## 1. 实验预设一键切换

`main.py` 顶部有一个 `EXPERIMENT_PRESET` 变量:

| 预设 | 模型 | Rounds | Clients/round | α | Batch | LR | Epochs | 用途 |
|---|---|---|---|---|---|---|---|---|
| `FEDRTS_RESNET18` | ResNet-18 | 500 | 10 (K=10) | 0.5 | 64 | 0.01 | 5 | 与 FedMef / FedRTS 对齐 |
| `HDM_RESNET20` | ResNet-20 | 200 | 20 (K=20) | 1.0 | 128 | 0.005 | 2 | 原始 HDM 配置 |

切换方式: 改一行字符串即可.

```python
EXPERIMENT_PRESET = "FEDRTS_RESNET18"   # 或 "HDM_RESNET20"
```

## 2. 只跑某一组策略

`main.py` 顶部还有 `RUN_ONLY`:

```python
RUN_ONLY = []                      # 空列表 = 全部 4 组都跑
RUN_ONLY = ["Magnitude"]           # 只跑 Magnitude
RUN_ONLY = ["None", "HDM"]         # 只跑 None + HDM
```

可用的组名: `None`, `Magnitude`, `Delta`, `HDM`.

## 3. 显存 / 并发

单个客户端的 GPU 额度通过 `CLIENT_NUM_GPUS` 控制, 影响 Ray 的最大并发 actor 数.

- 默认值 `0.1` 意味着一张 24GB 卡最多同时跑 10 个客户端.
- 每轮 sample 几个客户端就设成 `1 / K`, 让这些客户端一次装满同时跑 (最优).
- 如果显存占用远低于上限, 考虑: (a) 增大 `FRACTION_FIT` 让每轮客户端数更多, (b) 降低 `CLIENT_NUM_GPUS` 让并发更高, (c) 两者必须匹配.

典型组合:
| FRACTION_FIT | K | CLIENT_NUM_GPUS | 显存占用 (ResNet-18) |
|---|---|---|---|
| 0.1 | 10 | 0.1 | ~11 GB |
| 0.1 | 10 | 0.2 | ~6 GB |
| 0.2 | 20 | 0.05 | ~20 GB |

## 4. 运行训练

```bash
cd E:/Code/大创论文/paper/code
python main.py
```

训练过程中, 每一轮结束都会往 `records/{exp_name}_rounds.jsonl` 追加一行 JSON. 进程中途挂掉时前 N 轮的数据仍然保留.

实验正常结束后还会额外 dump `records/{exp_name}_full.pkl` (包含完整超参数+历史).

## 5. 记录文件格式

### 5.1 `{exp_name}_rounds.jsonl`
每行一轮, 字段:

```json
{"round": 1, "avg_fit_duration": 12.34, "avg_upload_size_kb": 5678.90, "loss": 2.1, "accuracy": 0.1234}
```

### 5.2 `{exp_name}_full.pkl`

```python
{
  "exp_name": "Magnitude",
  "strategy": "magnitude",
  "threshold": 0.70,
  "model": "ResNet18",
  "dataset": "CIFAR10",
  "num_rounds": 500,
  "alpha": 0.5,
  "fraction_fit": 0.1,
  "global_accuracies": [(round, acc), ...],
  "client_metrics": [{"round": ..., "avg_upload_size_kb": ..., "avg_fit_duration": ...}, ...],
}
```

**注意**: 每次调用 `recorder.start_experiment(exp_name)` 会清空同名 JSONL. 想保留老数据就先手动重命名备份.

## 6. 事后重绘图 (不重跑训练)

`plot_from_records.py` 独立于训练, 直接从 `records/` 读数据出图.

### 自动发现所有实验, 画对比图
```bash
python plot_from_records.py
# 输出: exp_comparison.png
```

### 只画单个实验
```bash
python plot_from_records.py Magnitude
# 输出: exp_magnitude_single.png
# 标题自动变成 "Magnitude Single Run" 而非 "Avg/Global"
```

### 画指定几个的对比图
```bash
python plot_from_records.py Magnitude HDM
```

### 截取前 N 轮
```bash
python plot_from_records.py --max-round 300
```

### 指定输出文件名
```bash
python plot_from_records.py Magnitude HDM --out paper_fig.png
```

### 优先读 pickle (默认优先读 JSONL, 因为 JSONL 实时更新)
```bash
python plot_from_records.py --pkl
```

## 7. 并行跑多个终端

可以. 每个终端各自写 `{exp_name}_*` 文件, 名字不冲突. 典型用法:

```bash
# 终端 A
RUN_ONLY = ["Magnitude"]   在 main.py 里
python main.py

# 终端 B (用不同的 main.py 副本, 或先启动 A 再改 RUN_ONLY 启动 B)
RUN_ONLY = ["HDM"]
python main.py
```

**注意事项**:
1. 两个终端共享同一张 GPU. 每个 Ray 集群都以为自己独占卡, 显存会叠加. 建议各自把 `CLIENT_NUM_GPUS` 调到 0.2, 总占用约 10 GB.
2. 首次下载 CIFAR-10 时两个终端会撞车. 先单独跑一个让它下完, 再并行跑.
3. `结果.txt` 文件名不带策略后缀, 两个终端互相覆盖. 如需保留, 手动备份或改代码为 `结果_{exp}.txt`.
4. 跑完后直接 `python plot_from_records.py` 就能一次画出两个策略的对比图.

## 8. 结果汇总 `结果.txt`

训练结束后写出 `结果.txt`, 包含:
- 数据集 / 模型 / 总参数量 / 客户端数 / 轮数 (元信息)
- Dense baseline (None 策略) 的平均上传 KB
- 每个实验: Final Acc, Best Acc, Last Upload (KB), Avg Upload (KB), Ratio (× dense)

`Ratio (× dense)` 是与 FedMef / FedRTS 论文中 `Data Exchange ratio` 跨论文对比的关键列.

参考数字 (CIFAR-10 + ResNet-18, α=0.5):
- FedDST: 0.232× @ 79.7%
- FedMef: 0.243× @ 81.7%
- FedRTS: 0.233× @ 73.41%

## 9. 常见故障排查

| 现象 | 原因 / 修复 |
|---|---|
| `CUDA out of memory` | 降低 `FRACTION_FIT` 或提高 `CLIENT_NUM_GPUS` |
| Ray actor 启动失败 | `ray.shutdown()` 没生效, 重启 Python 进程 |
| JSONL 有重复轮号 | 训练被打断又重跑; `recorder.start_experiment` 会覆盖, 但如果你绕过了 start 调用就会堆积. 手动删 jsonl 重跑即可 |
| `KeyError` in generate_plot | 某组实验没跑完; `plot_from_records.py` 会跳过缺失的组 |
| `None` 组的准确率很低 | 正常, `None` 是 dense FedAvg, 非 IID + 少量轮数下精度本来就不高 |

## 10. 改动与扩展

- **换数据集**: 把 `DATASET_NAME` 改成 `MNIST` / `CIFAR100` (后者需配合 100 类模型).
- **加新剪枝策略**: 在 `pruning_utils.py` 的 `apply_pruning` 里加一条分支, 然后在 `EXPERIMENTS_CONFIG` 里新增一项.
- **换模型**: `models.py` 里已有 ResNet-18/20, 可直接加 ResNet-34 等. `server.py` 和 `main.py` 的模型工厂都是按 `MODEL_NAME` 分派的, 加一行 `elif` 即可.
