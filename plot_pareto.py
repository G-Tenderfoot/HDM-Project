"""Generate a publication-quality Pareto frontier figure for the HDM paper.

Usage:
    python plot_pareto.py              # saves to ../figures/pareto.pdf
    python plot_pareto.py --out foo.pdf # custom output path
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -- Data --
# HDM configs: (ratio, best_acc, label)
HDM_POINTS = [
    (1.000, 90.73, "FedAvg (Dense)"),
    (0.494, 89.02, "HDM (0.75+3.0)"),
    (0.418, 87.82, "HDM (0.85+2.0)"),
    (0.376, 82.51, "HDM (0.88+2.0)"),
    (0.343, 83.39, "HDM (0.90+2.0)"),
    (0.271, 77.89, "HDM (0.92+2.5)"),
    (0.196, 78.49, "HDM (0.95+3.0)"),
]

# Literature baselines (same setup: ResNet-18, CIFAR-10, alpha=0.5, s_tm=0.8)
LIT_POINTS = [
    (0.232, 79.7,  "FedDST"),
    (0.243, 80.3,  "FedTiny"),
    (0.243, 81.7,  "FedMef"),
    (0.233, 73.41, "FedRTS"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "figures", "pareto.pdf"
    )

    fig, ax = plt.subplots(figsize=(6, 4.2))

    # -- HDM Pareto curve --
    hdm_x = [p[0] for p in HDM_POINTS]
    hdm_y = [p[1] for p in HDM_POINTS]
    order = np.argsort(hdm_x)[::-1]
    hdm_x_sorted = [hdm_x[i] for i in order]
    hdm_y_sorted = [hdm_y[i] for i in order]
    hdm_labels = [HDM_POINTS[i][2] for i in order]

    ax.plot(hdm_x_sorted, hdm_y_sorted, "o-", color="tab:purple", linewidth=2,
            markersize=7, label="HDM (ours)", zorder=5)

    # Annotate HDM points
    for x, y, label in zip(hdm_x_sorted, hdm_y_sorted, hdm_labels):
        if label == "FedAvg (Dense)":
            continue
        short = label.replace("HDM ", "")
        if "0.88" in label:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(-45, -8), fontsize=7, color="tab:purple")
        elif "0.95" in label:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(-50, 5), fontsize=7, color="tab:purple")
        elif "0.92" in label:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(8, -8), fontsize=7, color="tab:purple")
        else:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(8, 5), fontsize=7, color="tab:purple")

    # -- FedAvg point --
    ax.plot(1.0, 90.73, "s", color="tab:blue", markersize=9,
            label="FedAvg (Dense)", zorder=6)
    ax.annotate("FedAvg", (1.0, 90.73), textcoords="offset points",
                xytext=(-40, -12), fontsize=8, color="tab:blue")

    # -- Literature baselines --
    markers = {"FedDST": "^", "FedTiny": "v", "FedMef": "D", "FedRTS": "X"}
    colors  = {"FedDST": "tab:green", "FedTiny": "tab:cyan",
               "FedMef": "tab:orange", "FedRTS": "tab:red"}
    for x, y, name in LIT_POINTS:
        ax.plot(x, y, markers[name], color=colors[name], markersize=9,
                label=name, zorder=6)
        ax.annotate(f"{name}\n{y}%", (x, y), textcoords="offset points",
                    xytext=(8, -2), fontsize=7, color=colors[name])

    # -- Formatting --
    ax.set_xlabel(r"Communication Ratio ($\times$ dense)", fontsize=11)
    ax.set_ylabel("Best Accuracy (%)", fontsize=11)
    ax.set_xlim(0.15, 1.08)
    ax.set_ylim(70, 93)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("Accuracy vs. Communication Cost (ResNet-18, CIFAR-10)",
                 fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
