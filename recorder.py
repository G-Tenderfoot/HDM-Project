"""Per-round recording utility.

每个实验跑的时候往 records/{exp_name}_rounds.jsonl 每轮追加一行 JSON.
同一轮的 fit 指标 (upload/fit_duration) 和 eval 指标 (accuracy) 先后到达,
用 round 号做 merge, 最终每轮对应一个完整字典.

Flower 调用顺序 (每轮): configure_fit -> aggregate_fit -> evaluate_fn -> ...
所以先写 fit 指标, 再 merge accuracy.
"""
import json
import os
import threading
from typing import Optional

_RECORDS_DIR = None
_CURRENT_EXP: Optional[str] = None
_LOCK = threading.Lock()
# round -> merged dict (buffer, 用于合并 fit + eval 两次回调)
_ROUND_BUFFER = {}


def set_records_dir(path: str):
    global _RECORDS_DIR
    _RECORDS_DIR = path
    os.makedirs(path, exist_ok=True)


def start_experiment(exp_name: str):
    """在每个实验循环开始时调用一次. 清空 buffer, 准备好 jsonl 文件."""
    global _CURRENT_EXP, _ROUND_BUFFER
    with _LOCK:
        _CURRENT_EXP = exp_name
        _ROUND_BUFFER = {}
        path = _jsonl_path()
        if path and os.path.exists(path):
            os.remove(path)  # 重跑同名实验时覆盖


def _jsonl_path():
    if _RECORDS_DIR is None or _CURRENT_EXP is None:
        return None
    return os.path.join(_RECORDS_DIR, f"{_CURRENT_EXP}_rounds.jsonl")


def _flush_round(server_round: int):
    """如果该轮 buffer 同时有 fit 和 eval 指标, 写一行到 jsonl.
    或者等到 eval 回调时把之前记下的 fit 指标一起写.
    策略: 每次 update 都把当前 buffer 以覆盖方式写 (jsonl 实际会重复),
    改为: 只在 eval 回调时 flush, 因为 eval 在同一轮的 fit 之后.
    """
    pass  # 暂时不用


def record_fit_metrics(server_round: int, avg_fit_duration: float,
                       avg_upload_size_kb: float):
    """aggregate_fit 结束时调用."""
    with _LOCK:
        entry = _ROUND_BUFFER.setdefault(server_round, {"round": server_round})
        entry["avg_fit_duration"] = avg_fit_duration
        entry["avg_upload_size_kb"] = avg_upload_size_kb


def record_eval_metrics(server_round: int, loss: float, accuracy: float):
    """evaluate_global_model 结束时调用. 触发本轮写盘."""
    with _LOCK:
        entry = _ROUND_BUFFER.setdefault(server_round, {"round": server_round})
        entry["loss"] = loss
        entry["accuracy"] = accuracy

        path = _jsonl_path()
        if path is None:
            return
        # append 一行
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def read_jsonl(exp_name: str, records_dir: str):
    """读回一个实验的所有轮. 返回 list[dict], 按 round 排序."""
    path = os.path.join(records_dir, f"{exp_name}_rounds.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda x: x.get("round", 0))
    return rows
