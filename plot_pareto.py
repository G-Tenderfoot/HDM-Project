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
# Current bidirectional sparse upload+download results:
# (ExchangeRatio, best accuracy, label)
HDM_POINTS = [
    (1.0000, 90.73, "FedAvg (Dense)"),
    (0.3085, 87.81, "HDM (0.90+2.0)"),
    (0.2441, 87.09, "HDM (0.92+2.5)"),
    (0.1370, 84.72, "HDM (0.96+3.0)"),
    (0.1126, 81.60, "HDM (0.97+4.0)"),
    (0.1068, 79.05, "HDM (0.98+4.0)"),
    (0.0508, 75.78, "HDM (0.985+4.0)"),
]

# Literature baselines under the aligned ResNet-18/CIFAR-10 setup.
LIT_POINTS = [
    (0.086, 73.60, "FedMef 95%"),
    (0.138, 78.10, "FedMef 90%"),
    (0.243, 81.70, "FedMef 80%"),
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
        if "0.985" in label:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(7, -3), fontsize=7, color="tab:purple")
        elif "0.98" in label:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(7, 3), fontsize=7, color="tab:purple")
        elif "0.97" in label:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(8, -10), fontsize=7, color="tab:purple")
        elif "0.92" in label:
            ax.annotate("(0.92+2.5)", (x, y), textcoords="offset points",
                        xytext=(8, -8), fontsize=7, color="tab:purple")
        elif "0.96" in label:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(8, 6), fontsize=7, color="tab:purple")
        else:
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(8, 5), fontsize=7, color="tab:purple")

    # -- FedAvg point --
    ax.plot(1.0, 90.73, "s", color="tab:blue", markersize=9,
            label="FedAvg (Dense)", zorder=6)
    ax.annotate("FedAvg", (1.0, 90.73), textcoords="offset points",
                xytext=(-40, -12), fontsize=8, color="tab:blue")

    # -- Literature baselines --
    markers = {"FedMef 95%": "D", "FedMef 90%": "D",
               "FedMef 80%": "D", "FedRTS": "X"}
    colors  = {"FedMef 95%": "tab:orange", "FedMef 90%": "tab:orange",
               "FedMef 80%": "tab:orange", "FedRTS": "tab:red"}
    for x, y, name in LIT_POINTS:
        ax.plot(x, y, markers[name], color=colors[name], markersize=9,
                label=name, zorder=6)
        offset = (8, -2) if name != "FedRTS" else (8, -12)
        ax.annotate(f"{name}\n{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=offset, fontsize=7, color=colors[name])

    # -- Formatting --
    ax.set_xlabel(r"Communication Ratio ($\times$ dense)", fontsize=11)
    ax.set_ylabel("Best Accuracy (%)", fontsize=11)
    ax.set_xlim(0.03, 1.08)
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
