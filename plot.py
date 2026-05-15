import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.colors import LinearSegmentedColormap

# ===================== 配色 =====================
LINE_COLORS = ["#fbd2bc", "#feab88", "#b71c2c", "#8b0824", "#6a0624"]
HEATMAP_COLORS = [
    "#006633",
    "#339966",
    "#99cc66",
    "#ffffcc",
    "#ff9966",
    "#cc3333"
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("green_yellow_red", HEATMAP_COLORS)

# ===================== 路径 =====================
CHECKPOINT_PATH = "checkpoint/ablation_final.pkl"
RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================== 加载数据 =====================
with open(CHECKPOINT_PATH, 'rb') as f:
    data = pickle.load(f)

# ===================== 画图工具函数 =====================
def plot_curve(data_group, xs, y_type, title, xlabel, save_name, keys, labels, colors):
    plt.figure(figsize=(10, 5))
    markers = ["o", "s", "^", "D", "v"]
    for i, k in enumerate(keys):
        ys = [data_group[x][k][y_type] for x in xs]
        plt.plot(xs, ys, marker=markers[i], color=colors[i], 
                 label=labels[i], linewidth=3, markersize=9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(y_type.replace("_", " ").title())
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, save_name), dpi=200)
    plt.close()

# ===================== 消融实验绘图 =====================
def plot_ablation(data):
    plt.rcParams.update({'font.size': 12})

    # -------------------- 算法与颜色配置 --------------------
    all_keys   = ["MaxEnt", "BT", "MaxEntBT"]
    all_labels = ["MaxEnt", "BT", "MaxEnt+BT"]
    all_colors = ["#feab88", "#b71c2c", "#8b0824"]

    pref_keys  = ["BT", "MaxEntBT"]
    pref_labels = ["BT", "MaxEnt+BT"]
    pref_colors = ["#b71c2c", "#8b0824"]

    # -------------------- 1. 噪声消融 --------------------
    xs = [0, 0.1, 0.2, 0.3]
    plot_curve(data["noise"], xs, "success_rate", "Noise Robustness - Success Rate", "Noise", "noise_sr.png",
               all_keys, all_labels, all_colors)
    plot_curve(data["noise"], xs, "sim", "Noise Robustness - Similarity", "Noise", "noise_sim.png",
               all_keys, all_labels, all_colors)

    # -------------------- 2. 轨迹数消融 --------------------
    xs = [20, 30, 50, 100]
    plot_curve(data["traj"], xs, "success_rate", "Trajectory Count - Success Rate", "Num Trajectories", "traj_sr.png",
               all_keys, all_labels, all_colors)
    plot_curve(data["traj"], xs, "sim", "Trajectory Count - Similarity", "Num Trajectories", "traj_sim.png",
               all_keys, all_labels, all_colors)

    # -------------------- 3. 网格尺寸消融 --------------------
    xs = [6, 8, 10]
    plot_curve(data["grid"], xs, "success_rate", "Grid Size - Success Rate", "Size", "grid_sr.png",
               all_keys, all_labels, all_colors)
    plot_curve(data["grid"], xs, "sim", "Grid Size - Similarity", "Size", "grid_sim.png",
               all_keys, all_labels, all_colors)

    # -------------------- 4. 偏好数消融 --------------------
    xs = [100, 200, 300, 500]
    plot_curve(data["prefs"], xs, "success_rate", "Preference Count - Success Rate", "Num Preferences", "pref_sr.png",
               pref_keys, pref_labels, pref_colors)
    plot_curve(data["prefs"], xs, "sim", "Preference Count - Similarity", "Num Preferences", "pref_sim.png",
               pref_keys, pref_labels, pref_colors)

# ===================== 热力图 =====================
def plot_all_heatmaps(data):
    sizes = [6, 8, 10]
    titles = ["GT", "MaxEnt", "BT", "MaxEnt+BT"]
    keys = ["GT", "MaxEnt", "BT", "MaxEntBT"]

    for sz in sizes:
        res = data["grid"][sz]
        rs = [res[k]["reward"] for k in keys]
        plt.figure(figsize=(18, 4))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            r = rs[i].astype(np.float64)
            r_min, r_max = r.min(), r.max()
            r_norm = (r - r_min) / (r_max - r_min + 1e-8)
            im = plt.imshow(r_norm, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
            plt.title(f"{titles[i]} (size={sz})", fontsize=13)
            plt.axis("off")
            plt.colorbar(im, fraction=0.045, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"result/heatmap_{sz}.png", dpi=200, bbox_inches="tight")
        plt.close()

# ===================== 主函数 =====================
if __name__ == "__main__":
    print("Generating ablation plots and heatmaps...")
    plot_ablation(data)
    plot_all_heatmaps(data)
    print(f"Done! All figures saved to: {RESULT_DIR}")
