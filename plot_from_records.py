"""Regenerate plots from records/ without retraining.

用法 (在 code/ 目录下):
    python plot_from_records.py                       # 自动加载 records/ 里所有 {exp}_rounds.jsonl 或 {exp}_full.pkl, 画出对比图
    python plot_from_records.py Magnitude             # 只画 Magnitude 单实验图
    python plot_from_records.py Magnitude HDM         # 画 Magnitude+HDM 对比图
    python plot_from_records.py --pkl                 # 优先使用 .pkl (更完整), 默认优先 .jsonl (增量写盘, 更新鲜)
    python plot_from_records.py --max-round 300       # 只画到第 300 轮
    python plot_from_records.py --out custom.png      # 自定义输出文件名

记录文件约定:
    records/{exp_name}_rounds.jsonl  每轮一行: {round, accuracy, loss, avg_upload_size_kb, avg_fit_duration}
    records/{exp_name}_full.pkl      实验结束后 dump 的完整结果
"""
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt

import recorder

RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records")

# 与 main.py EXPERIMENTS_CONFIG 保持一致的颜色
COLOR_MAP = {
    "None": "tab:blue",
    "Magnitude": "tab:orange",
    "Delta": "tab:green",
    "HDM": "tab:purple",
}


def discover_experiments(records_dir: str):
    """列出 records_dir 下所有有记录的实验名."""
    names = set()
    if not os.path.isdir(records_dir):
        return []
    for fname in os.listdir(records_dir):
        if fname.endswith("_rounds.jsonl"):
            names.add(fname[: -len("_rounds.jsonl")])
        elif fname.endswith("_full.pkl"):
            names.add(fname[: -len("_full.pkl")])
    return sorted(names)


def load_exp(name: str, records_dir: str, prefer_pkl: bool = False):
    """返回 (global_accuracies=[(round, acc)], client_metrics=[{round, avg_upload_size_kb, avg_fit_duration}])

    - 优先读 jsonl (增量写入, 更新鲜), 除非 prefer_pkl=True
    - jsonl 里每行同时含 accuracy/upload/fit, 所以可以直接拆成两种格式
    - pkl 格式是 Flower history.metrics_centralized["accuracy"] (list[(round, acc)]) + strategy.metrics_aggregated_per_round
    """
    pkl_path = os.path.join(records_dir, f"{name}_full.pkl")
    jsonl_path = os.path.join(records_dir, f"{name}_rounds.jsonl")

    use_pkl = prefer_pkl and os.path.exists(pkl_path)
    if (not use_pkl) and os.path.exists(jsonl_path):
        rows = recorder.read_jsonl(name, records_dir)
        if not rows:
            # jsonl 空了, 回退 pkl
            use_pkl = os.path.exists(pkl_path)
        else:
            accs = [(r["round"], r["accuracy"]) for r in rows if "accuracy" in r]
            metrics = [
                {
                    "round": r["round"],
                    "avg_upload_size_kb": r.get("avg_upload_size_kb", 0.0),
                    "avg_fit_duration": r.get("avg_fit_duration", 0.0),
                }
                for r in rows
            ]
            return accs, metrics

    if use_pkl or os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        accs = list(data.get("global_accuracies", []))
        metrics = list(data.get("client_metrics", []))
        return accs, metrics

    return [], []


def generate_plot(exp_data: dict, title: str, filename: str, max_round: int = None):
    """exp_data: {name: (accs, metrics)}"""
    names = [n for n in exp_data if exp_data[n][0] or exp_data[n][1]]
    if not names:
        print("No data to plot.")
        return
    is_single = len(names) == 1
    only_name = names[0] if is_single else None

    plt.figure(figsize=(20, 6))

    # Subplot 1: Accuracy
    plt.subplot(1, 3, 1)
    for name in names:
        accs, _ = exp_data[name]
        if not accs:
            continue
        rounds = [r for r, _ in accs]
        values = [a for _, a in accs]
        if max_round is not None:
            pairs = [(r, a) for r, a in zip(rounds, values) if r <= max_round]
            rounds = [r for r, _ in pairs]
            values = [a for _, a in pairs]
        plt.plot(rounds, values, label=name, color=COLOR_MAP.get(name), linewidth=2)
    plt.title(f"{only_name} Accuracy" if is_single else "Global Accuracy")
    plt.xlabel("Rounds"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()

    # Subplot 2: Upload Size
    plt.subplot(1, 3, 2)
    for name in names:
        _, metrics = exp_data[name]
        if not metrics:
            continue
        rounds = [m["round"] for m in metrics]
        uploads = [m.get("avg_upload_size_kb", 0.0) for m in metrics]
        if max_round is not None:
            pairs = [(r, u) for r, u in zip(rounds, uploads) if r <= max_round]
            rounds = [r for r, _ in pairs]
            uploads = [u for _, u in pairs]
        plt.plot(rounds, uploads, label=name, color=COLOR_MAP.get(name), linewidth=2)
    plt.title(f"{only_name} Upload Size per Client (KB)" if is_single else "Avg Upload Size (KB)")
    plt.xlabel("Rounds"); plt.grid(True); plt.legend()

    # Subplot 3: Fit Duration
    plt.subplot(1, 3, 3)
    for name in names:
        _, metrics = exp_data[name]
        if not metrics:
            continue
        rounds = [m["round"] for m in metrics]
        times = [m.get("avg_fit_duration", 0.0) for m in metrics]
        if max_round is not None:
            pairs = [(r, t) for r, t in zip(rounds, times) if r <= max_round]
            rounds = [r for r, _ in pairs]
            times = [t for _, t in pairs]
        plt.plot(rounds, times, label=name, color=COLOR_MAP.get(name), linewidth=2)
    plt.title(f"{only_name} Fit Duration per Client (s)" if is_single else "Avg Fit Duration (s)")
    plt.xlabel("Rounds"); plt.grid(True); plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")


def generate_summary(exp_data: dict, records_dir: str, out_path: str):
    """从 records 重建 结果.txt, 用 None 作 dense baseline."""
    # 计算 dense baseline
    dense_baseline_kb = 0.0
    if "None" in exp_data:
        _, none_metrics = exp_data["None"]
        none_kbs = [m.get("avg_upload_size_kb", 0.0) for m in none_metrics]
        if none_kbs:
            dense_baseline_kb = sum(none_kbs) / len(none_kbs)
    if dense_baseline_kb <= 0.0:
        all_kbs = []
        for name in exp_data:
            _, metrics = exp_data[name]
            if metrics:
                all_kbs.append(max(m.get("avg_upload_size_kb", 0.0) for m in metrics))
        dense_baseline_kb = max(all_kbs) if all_kbs else 1.0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("实验结果汇总 (Experiment Summary) - 从 records 重建\n")
        f.write("========================================\n")
        f.write(f"Records dir: {records_dir}\n")
        f.write(f"Dense baseline (avg upload KB): {dense_baseline_kb:.2f}\n")
        f.write("Ratio = avg_upload_kb / dense_baseline_kb\n")
        f.write("========================================\n\n")

        headers = ["Experiment", "Final Acc", "Best Acc",
                   "Last Upload(KB)", "Avg Upload(KB)", "Ratio(x dense)"]
        header_str = (f"{headers[0]:<12} | {headers[1]:<10} | {headers[2]:<10} | "
                      f"{headers[3]:<15} | {headers[4]:<15} | {headers[5]:<15}")
        f.write(header_str + "\n")
        f.write("-" * len(header_str) + "\n")

        # 按固定顺序输出
        order = ["None", "Magnitude", "Delta", "HDM"]
        for name in order:
            if name not in exp_data:
                continue
            accs, metrics = exp_data[name]
            accuracies = [a for _, a in accs]
            final_acc = accuracies[-1] if accuracies else 0.0
            best_acc = max(accuracies) if accuracies else 0.0

            all_kbs = [m.get("avg_upload_size_kb", 0.0) for m in metrics]
            last_upload = all_kbs[-1] if all_kbs else 0.0
            avg_upload = sum(all_kbs) / len(all_kbs) if all_kbs else 0.0
            ratio = avg_upload / dense_baseline_kb if dense_baseline_kb > 0 else 0.0

            row = (f"{name:<12} | {final_acc:<10.4f} | {best_acc:<10.4f} | "
                   f"{last_upload:<15.2f} | {avg_upload:<15.2f} | {ratio:<15.4f}")
            f.write(row + "\n")

        # 也输出不在 order 里的实验
        for name in exp_data:
            if name in order:
                continue
            accs, metrics = exp_data[name]
            accuracies = [a for _, a in accs]
            final_acc = accuracies[-1] if accuracies else 0.0
            best_acc = max(accuracies) if accuracies else 0.0
            all_kbs = [m.get("avg_upload_size_kb", 0.0) for m in metrics]
            last_upload = all_kbs[-1] if all_kbs else 0.0
            avg_upload = sum(all_kbs) / len(all_kbs) if all_kbs else 0.0
            ratio = avg_upload / dense_baseline_kb if dense_baseline_kb > 0 else 0.0
            row = (f"{name:<12} | {final_acc:<10.4f} | {best_acc:<10.4f} | "
                   f"{last_upload:<15.2f} | {avg_upload:<15.2f} | {ratio:<15.4f}")
            f.write(row + "\n")

    print(f"Summary saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="Experiment names to plot. Omit to auto-discover.")
    parser.add_argument("--pkl", action="store_true", help="Prefer pickle over jsonl.")
    parser.add_argument("--max-round", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output filename. Auto-named if omitted.")
    parser.add_argument("--records-dir", default=RECORDS_DIR)
    parser.add_argument("--summary", action="store_true",
                        help="Generate 结果.txt from records instead of plotting.")
    parser.add_argument("--summary-out", default=None,
                        help="Output path for --summary (default: 结果.txt in records dir).")
    args = parser.parse_args()

    names = args.names or discover_experiments(args.records_dir)
    if not names:
        print(f"No records found in {args.records_dir}.")
        sys.exit(1)

    exp_data = {}
    for name in names:
        accs, metrics = load_exp(name, args.records_dir, prefer_pkl=args.pkl)
        if not accs and not metrics:
            print(f"Warning: no data for {name}, skipping.")
            continue
        exp_data[name] = (accs, metrics)

    if not exp_data:
        sys.exit(1)

    # --summary 模式: 生成结果.txt
    if args.summary:
        summary_path = args.summary_out or os.path.join(args.records_dir, "结果.txt")
        generate_summary(exp_data, args.records_dir, summary_path)
        return

    # 绘图模式
    if args.out:
        filename = args.out
    elif len(exp_data) == 1:
        only = next(iter(exp_data))
        filename = f"exp_{only.lower()}_single.png"
    else:
        filename = "exp_comparison.png"

    if len(exp_data) == 1:
        only = next(iter(exp_data))
        max_r = args.max_round or (max((r for r, _ in exp_data[only][0]), default=0) or 0)
        title = f"{only} Single Run (Round 0-{max_r})"
    else:
        max_r = args.max_round or max(
            (max((r for r, _ in accs), default=0) for accs, _ in exp_data.values()),
            default=0,
        )
        title = f"Pruning Strategy Comparison (Round 0-{max_r})"

    generate_plot(exp_data, title, filename, max_round=args.max_round)


if __name__ == "__main__":
    main()
