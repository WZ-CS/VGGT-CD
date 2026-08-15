"""
VGGT-CD Pose Evaluation Script (v5)
GT：浅灰散点
Pred：蓝色散点
无渐变，无T1/T2分界，无菱形
加入时间（s）和显存（MB）输出
针对论文需求优化了排版、字体和画质
"""
import os, json, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def umeyama_sim3(src, dst):
    N = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc**2).sum() / N
    cov   = (dc.T @ sc) / N
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3); S[2,2] = np.linalg.det(U @ Vt)
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t

def apply_sim3(pts, s, R, t):
    return s * (R @ pts.T).T + t

def pose44_to_center(T):
    return -T[:3,:3].T @ T[:3,3]

def poses_to_centers(poses):
    return np.array([pose44_to_center(p) for p in poses])

def relative_pose(Ti, Tj):
    return Tj @ np.linalg.inv(Ti)

def rot_angle_deg(R):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(R)-1.0)/2.0, -1.0, 1.0))))


def eval_joint(pred_t1, pred_t2, gt_t1, gt_t2, fine_s, fine_R, fine_t):
    N1 = len(pred_t1)
    c_t1_in_t2 = apply_sim3(poses_to_centers(pred_t1), fine_s, fine_R, fine_t)
    pred_joint  = np.concatenate([c_t1_in_t2, poses_to_centers(pred_t2)], axis=0)
    gt_joint    = np.concatenate([poses_to_centers(gt_t1), poses_to_centers(gt_t2)], axis=0)

    s_a, R_a, t_a = umeyama_sim3(pred_joint, gt_joint)
    aligned = apply_sim3(pred_joint, s_a, R_a, t_a)
    ate_per = np.linalg.norm(aligned - gt_joint, axis=1)

    # RTE/RRE：T1和T2分别计算相邻帧误差，跳过拼接处（T1末帧→T2首帧不是真实相邻）
    rte_list, rre_list = [], []
    for seq_pred, seq_gt in [(pred_t1, gt_t1), (pred_t2, gt_t2)]:
        for i in range(len(seq_pred) - 1):
            Tgt   = relative_pose(seq_gt[i],   seq_gt[i+1])
            Tpred = relative_pose(seq_pred[i], seq_pred[i+1])
            rte_list.append(np.linalg.norm(s_a * Tpred[:3,3] - Tgt[:3,3]))
            rre_list.append(rot_angle_deg(Tgt[:3,:3] @ Tpred[:3,:3].T))
    N = len(pred_t1) + len(pred_t2)

    m = dict(
        N          = N,
        ATE_mean   = round(float(ate_per.mean()),               5),
        ATE_rmse   = round(float(np.sqrt((ate_per**2).mean())), 5),
        ATE_median = round(float(np.median(ate_per)),           5),
        RTE_mean   = round(float(np.mean(rte_list)),            5),
        RRE_mean   = round(float(np.mean(rre_list)),            4),
    )
    return m, aligned, gt_joint, ate_per


def plot_final(aligned, gt, metrics, scene_name, save_path):
    # 设置论文发表级绘图参数
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 18
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    
    # 优化标题排版，拉开间距
    fig.suptitle(
        f"{scene_name}  |  "
        f"ATE = {metrics['ATE_mean']:.4f} m  |  "
        f"RMSE = {metrics['ATE_rmse']:.4f} m  |  "
        f"RTE = {metrics['RTE_mean']:.4f} m  |  "
        f"RRE = {metrics['RRE_mean']:.3f}°",
        fontweight="bold"
    )

    for ax, xi, yi, xlabel, ylabel, vtitle in [
        (ax1, 0, 1, "X (m)", "Y (m)", "Top View (XY)"),
        (ax2, 0, 2, "X (m)", "Z (m)", "Side View (XZ)"),
    ]:
        # 增大GT散点尺寸，颜色加深以便看清
        ax.scatter(gt[:,xi],      gt[:,yi],
                   c="#A0A0A0", s=45, alpha=0.7,
                   label="GT", zorder=2, edgecolors="none")
        
        # 增大Pred散点尺寸，并添加白色描边提升清晰度
        ax.scatter(aligned[:,xi], aligned[:,yi],
                   c="#2196F3", s=45, alpha=0.9,
                   label="Pred", zorder=3, edgecolors="white", linewidths=0.6)
        
        ax.set_xlabel(xlabel, fontweight="medium")
        ax.set_ylabel(ylabel, fontweight="medium")
        ax.set_title(vtitle, fontweight="medium", pad=12)
        ax.set_aspect("equal", adjustable="datalim")
        
        # 改用更适合学术图表的半透明虚线网格
        ax.grid(True, linestyle="--", alpha=0.6, lw=1.0)
        
        # 加粗坐标轴边框和刻度线
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5, length=5)
        
        ax.legend(loc="best", framealpha=0.9, edgecolor="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    # 输出 300 DPI 高清图像
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_summary_bar(rows, save_path):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })
    
    scenes = [r["scene"] for r in rows]
    n = len(scenes); x = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(max(n*2+2, 14), 5.5))
    fig.suptitle("VGGT-CD Evaluation Summary", fontsize=18, fontweight="bold")
    
    for ax, (key, color, ylabel, title) in zip(axes, [
        ("ATE_Joint", "#55A868", "ATE (m)",   "Absolute Translation Error"),
        ("RTE_mean",  "#4C72B0", "RTE (m)",   "Relative Translation Error"),
        ("RRE_mean",  "#C44E52", "RRE (deg)", "Relative Rotation Error"),
    ]):
        vals = [r.get(key, 0) for r in rows]
        # 添加黑色描边使柱状图更硬朗
        ax.bar(x, vals, 0.55, color=color, alpha=0.85, edgecolor='black', linewidth=0.8)
        
        for xi, y in enumerate(vals):
            ax.text(xi, y + (max(vals)*0.01), f"{y:.4f}", ha="center", va="bottom", fontsize=11)
            
        ax.set_xticks(x)
        ax.set_xticklabels(scenes, rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontweight="medium")
        ax.set_title(title, fontweight="medium", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.6, lw=1.0)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5, length=5)
        
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary chart: {save_path}")


def eval_scene(scene_dir):
    def npy(name):
        p = os.path.join(scene_dir, name)
        if not os.path.exists(p): raise FileNotFoundError(p)
        return np.load(p)

    pred_t1 = npy("pred_poses_t1.npy")
    pred_t2 = npy("pred_poses_t2.npy")
    gt_t1   = npy("gt_poses_t1.npy")
    gt_t2   = npy("gt_poses_t2.npy")

    sim3_path = os.path.join(scene_dir, "fine_sim3.npz")
    if not os.path.exists(sim3_path):
        raise FileNotFoundError(f"fine_sim3.npz not found in {scene_dir}")
    sim3   = np.load(sim3_path)
    fine_s = float(sim3["s"])
    fine_R = sim3["R"]
    fine_t = sim3["t"]
    
    # 尝试加载时间与显存信息
    time_s = 0.0
    mem_mb = 0.0
    timing_path = os.path.join(scene_dir, "timing.json")
    if os.path.exists(timing_path):
        with open(timing_path, "r") as f:
            timing_data = json.load(f)
            time_s = timing_data.get("paper_eval_time", 0.0)
            mem_mb = timing_data.get("peak_memory_MB", 0.0)

    scene_name = os.path.basename(scene_dir.rstrip("/"))
    N1, N2 = len(pred_t1), len(pred_t2)
    print(f"\n{'='*55}")
    print(f"Scene: {scene_name}   T1={N1}  T2={N2}  total={N1+N2}")
    print(f"fine_sim3 scale={fine_s:.4f}")

    metrics, aligned, gt_joint, ate_per = eval_joint(
        pred_t1, pred_t2, gt_t1, gt_t2, fine_s, fine_R, fine_t
    )
    print(f"  ATE={metrics['ATE_mean']:.4f}m  RMSE={metrics['ATE_rmse']:.4f}m  "
          f"RTE={metrics['RTE_mean']:.4f}m  RRE={metrics['RRE_mean']:.3f}deg")
    print(f"  Time={time_s:.4f}s  Mem={mem_mb:.2f}MB")

    # 更新 metrics 输出，包含时间和显存
    full_metrics = {
        "scene": scene_name, 
        **metrics, 
        "Time_s": time_s, 
        "Memory_MB": mem_mb
    }
    with open(os.path.join(scene_dir, "metrics.json"), "w") as f:
        json.dump(full_metrics, f, indent=2)

    plot_final(aligned, gt_joint, metrics, scene_name,
               save_path=os.path.join(scene_dir, "traj_final.png"))

    return {
        "scene":     scene_name,
        "N_frames":  N1 + N2,
        "ATE_Joint": metrics["ATE_mean"],
        "ATE_rmse":  metrics["ATE_rmse"],
        "RTE_mean":  metrics["RTE_mean"],
        "RRE_mean":  metrics["RRE_mean"],
        "Time(s)":   time_s,
        "Mem(MB)":   mem_mb,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", required=True)
    p.add_argument("--scene",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted([
            d for d in os.listdir(args.eval_root)
            if os.path.isdir(os.path.join(args.eval_root, d))
            and os.path.exists(os.path.join(args.eval_root, d, "pred_poses_t1.npy"))
        ])
        print(f"Found {len(scenes)} scenes: {scenes}")

    rows = []
    for sc in scenes:
        try:
            rows.append(eval_scene(os.path.join(args.eval_root, sc)))
        except Exception as e:
            import traceback
            print(f"ERROR {sc}: {e}"); traceback.print_exc()

    if not rows: print("No results."); return

    csv_path = os.path.join(args.eval_root, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # 打印汇总表格时加入 Time 和 Mem
    print("\n" + "="*80)
    print(f"{'Scene':<22} {'N':>5} {'ATE':>8} {'RMSE':>8} {'RTE':>8} {'RRE':>8} {'Time(s)':>8} {'Mem(MB)':>8}")
    print("-"*80)
    for r in rows:
        print(f"{r['scene']:<22} {r['N_frames']:>5} "
              f"{r['ATE_Joint']:>8.4f} {r['ATE_rmse']:>8.4f} "
              f"{r['RTE_mean']:>8.4f} {r['RRE_mean']:>8.3f} "
              f"{r['Time(s)']:>8.4f} {r['Mem(MB)']:>8.2f}")
    if len(rows) > 1:
        keys = ["ATE_Joint", "ATE_rmse", "RTE_mean", "RRE_mean", "Time(s)", "Mem(MB)"]
        means = {k: np.mean([r[k] for r in rows]) for k in keys}
        print("-"*80)
        print(f"{'MEAN':<22} {'':>5} "
              f"{means['ATE_Joint']:>8.4f} {means['ATE_rmse']:>8.4f} "
              f"{means['RTE_mean']:>8.4f} {means['RRE_mean']:>8.3f} "
              f"{means['Time(s)']:>8.4f} {means['Mem(MB)']:>8.2f}")
        plot_summary_bar(rows, os.path.join(args.eval_root, "summary.png"))
    print("="*80)


if __name__ == "__main__":
    main()