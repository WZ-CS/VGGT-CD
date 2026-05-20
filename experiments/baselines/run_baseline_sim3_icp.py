"""
VGGT-CD Baseline: VGGT + Sim3 ICP (Scale ICP)
===============================================
此 Baseline 专门用于对比解决"尺度歧义"的传统方法。

Pipeline:
  1. VGGT 独立稠密推理 T1 / T2
  2. 初始化: 质心对齐 + 包围盒对角线比例估算初始尺度 (Coarse)
  3. Sim3 ICP: 迭代寻找最近邻，并使用 Umeyama 算法同时优化 s, R, t (Fine)
  
输出格式与 run_inference.py 完全一致，可直接用 eval_poses_1.py 评估。

用法：
    python run_baseline_sim3_icp.py \
        --data_root /path/to/WAT \
        --scene car_resized \
        --seq_t1 IMG_2285 \
        --seq_t2 IMG_2286 \
        --output_root ./eval_results_sim3_icp
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import json
import time
import argparse
import struct
import collections
import glob
import gc
import numpy as np
import torch
from scipy.spatial import cKDTree

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vggt"))

from coarse_to_fine_registration import (
    umeyama_sim3,
    apply_sim3,
    align_intrinsics_and_depth,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# ===========================================================================
# 递归卸载工具（防显存泄漏）
# ===========================================================================
def recursive_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return arr
    elif isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_cpu(v) for v in obj)
    return obj


# ===========================================================================
# COLMAP reader
# ===========================================================================
CameraModel = collections.namedtuple("CameraModel", ["model_id","model_name","num_params"])
Camera      = collections.namedtuple("Camera",      ["id","model","width","height","params"])
Image       = collections.namedtuple("Image",       ["id","qvec","tvec","camera_id","name","xys","point3D_ids"])

CAMERA_MODELS = {
    0: CameraModel(0,"SIMPLE_PINHOLE",3),
    1: CameraModel(1,"PINHOLE",4),
    2: CameraModel(2,"SIMPLE_RADIAL",4),
    3: CameraModel(3,"RADIAL",5),
    4: CameraModel(4,"OPENCV",8),
}

def read_images_binary(path):
    images = {}
    with open(path,"rb") as f:
        num = struct.unpack("<Q",f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i",f.read(4))[0]
            qvec   = struct.unpack("<4d",f.read(32))
            tvec   = struct.unpack("<3d",f.read(24))
            cam_id = struct.unpack("<i",f.read(4))[0]
            name   = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            name = name.decode()
            num_pts     = struct.unpack("<Q",f.read(8))[0]
            xys         = struct.unpack(f"<{2*num_pts}d",f.read(16*num_pts))
            point3D_ids = struct.unpack(f"<{num_pts}q",  f.read(8*num_pts))
            images[img_id] = Image(img_id,qvec,tvec,cam_id,name,
                                   np.array(xys).reshape(-1,2),
                                   np.array(point3D_ids))
    return images

def qvec2rotmat(qvec):
    qw,qx,qy,qz = qvec
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qw*qz),  2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz),1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx),1-2*(qx**2+qy**2)],
    ])

def colmap_image_to_pose44(img):
    R = qvec2rotmat(img.qvec)
    t = np.array(img.tvec)
    T = np.eye(4,dtype=np.float64)
    T[:3,:3] = R; T[:3,3] = t
    return T

def load_colmap_poses(colmap_dir):
    pose_dict = {}
    for img in read_images_binary(os.path.join(colmap_dir,"images.bin")).values():
        pose_dict[os.path.basename(img.name)] = colmap_image_to_pose44(img)
    return pose_dict

def load_sequence(seq_dir, pose_dict):
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    all_imgs = sorted([p for p in glob.glob(os.path.join(seq_dir,"*"))
                       if os.path.splitext(p)[1] in exts])
    image_paths, gt_poses = [], []
    for p in all_imgs:
        bn = os.path.basename(p)
        if bn in pose_dict:
            image_paths.append(p)
            gt_poses.append(pose_dict[bn])
        else:
            print(f"  [warn] no GT pose for {bn}, skipping")
    print(f"  Loaded {len(image_paths)}/{len(all_imgs)} images with GT from {seq_dir}")
    return image_paths, np.array(gt_poses, dtype=np.float64)

# ===========================================================================
# VGGT 推理
# ===========================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    print("Loading VGGT model...")
    m = VGGT()
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.eval().to(device)
    print("Model loaded.")
    return m

def run_vggt_inference(image_paths, model):
    images = load_and_preprocess_images(image_paths).to(device)
    dtype  = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions = recursive_to_cpu(predictions)

    intrinsic_ref = predictions["intrinsic"][0]
    intrinsic_aligned, depth_aligned = align_intrinsics_and_depth(
        predictions["intrinsic"], intrinsic_ref, predictions["depth"][..., 0])
    predictions["intrinsic"] = intrinsic_aligned

    world_points = unproject_depth_map_to_point_map(
        depth_aligned[..., None], predictions["extrinsic"], intrinsic_aligned)
    predictions["world_points_from_depth"] = world_points

    del images; torch.cuda.empty_cache()
    return predictions

def to_44(ext34):
    N = ext34.shape[0]
    T = np.zeros((N,4,4),dtype=np.float64)
    T[:,:3,:] = ext34.astype(np.float64)
    T[:,3,3]  = 1.0
    return T

# ===========================================================================
# Sim3 ICP (Scale ICP) 实现
# ===========================================================================
def extract_high_conf_points(predictions, conf_percentile, max_points=100000):
    points = predictions["world_points_from_depth"]
    conf   = predictions["depth_conf"]
    thresh = np.percentile(conf[conf > 0], conf_percentile) if np.any(conf > 0) else 0.01
    valid  = conf.reshape(-1) > thresh
    pts = points.reshape(-1, 3)[valid].astype(np.float64)
    
    # 随机降采样加速 ICP，Sim3 ICP 是全局收敛性质的，点数不用特别多
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts

def scale_icp_registration(source_pts, target_pts, max_iter=50, tol=1e-5):
    """
    自定义的带尺度优化 ICP (Sim3 ICP) 算法
    """
    print(f"  [Sim3 ICP] Source pts: {len(source_pts):,} | Target pts: {len(target_pts):,}")
    t0 = time.time()
    
    # --- 1. 全局初始化 (Coarse Alignment) ---
    # 估算尺度：通过包围盒对角线的比例进行初始放缩
    bbox_src = source_pts.max(axis=0) - source_pts.min(axis=0)
    bbox_tgt = target_pts.max(axis=0) - target_pts.min(axis=0)
    init_s = float(np.linalg.norm(bbox_tgt) / max(np.linalg.norm(bbox_src), 1e-6))
    
    # 估算平移：质心对齐
    mu_s = source_pts.mean(axis=0)
    mu_t = target_pts.mean(axis=0)
    init_t = mu_t - init_s * mu_s
    init_R = np.eye(3)
    
    s, R, t = init_s, init_R, init_t
    
    # --- 2. 建立 KDTree 加速最近邻搜索 ---
    tgt_tree = cKDTree(target_pts)
    
    # 动态距离阈值 (类似 RANSAC 的 Inlier 阈值)，随着迭代逐步收紧
    max_dist = np.linalg.norm(bbox_tgt) * 0.2 
    prev_error = float('inf')
    
    for i in range(max_iter):
        # 将当前的 s, R, t 应用于源点云
        curr_src = apply_sim3(source_pts, s, R, t)
        
        # 寻找最近邻
        distances, indices = tgt_tree.query(curr_src, k=1)
        
        # 过滤掉距离过远的错误匹配对 (Outlier Rejection)
        valid = distances < max_dist
        if np.sum(valid) < 100:  # 匹配点太少，提早退出
            print(f"  [Sim3 ICP] Iter {i}: Too few correspondences, breaking.")
            break
            
        src_matched = source_pts[valid]  # 注意：必须使用最原始的点云代入 Umeyama 求解！
        tgt_matched = target_pts[indices[valid]]
        
        # 计算带尺度的刚体变换 (Umeyama Sim3)
        s_new, R_new, t_new = umeyama_sim3(src_matched, tgt_matched, with_scale=True)
        
        # 更新状态
        s, R, t = s_new, R_new, t_new
        
        # 检查收敛状态
        mean_error = np.mean(distances[valid])
        max_dist = max(mean_error * 2.5, np.linalg.norm(bbox_tgt) * 0.01) # 收紧阈值
        
        if abs(prev_error - mean_error) < tol:
            print(f"  [Sim3 ICP] Converged at iteration {i+1}.")
            break
        prev_error = mean_error

    elapsed = time.time() - t0
    print(f"  [Sim3 ICP] Done. t={elapsed:.3f}s, Final Scale={s:.4f}, Mean Error={prev_error:.4f}")
    
    info = {
        "sim3_icp_iterations": i + 1,
        "sim3_icp_error": float(prev_error),
        "t_sim3_icp": elapsed
    }
    return s, R, t, info

# ===========================================================================
# 主流程
# ===========================================================================

def run_scene_sim3_icp(scene_dir, seq_t1, seq_t2, output_dir, model, args):
    os.makedirs(output_dir, exist_ok=True)
    timing = {}

    if torch.cuda.is_available():
        import gc; gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- GT ----
    colmap_dir = os.path.join(scene_dir, "sparse", "0")
    pose_dict  = load_colmap_poses(colmap_dir)
    paths_t1, gt_t1 = load_sequence(os.path.join(scene_dir,"images",seq_t1), pose_dict)
    paths_t2, gt_t2 = load_sequence(os.path.join(scene_dir,"images",seq_t2), pose_dict)

    if len(paths_t1) == 0 or len(paths_t2) == 0:
        print("ERROR: no images with GT, skipping."); return

    with open(os.path.join(output_dir,"image_names_t1.txt"),"w") as f:
        f.writelines(p+"\n" for p in paths_t1)
    with open(os.path.join(output_dir,"image_names_t2.txt"),"w") as f:
        f.writelines(p+"\n" for p in paths_t2)

    np.save(os.path.join(output_dir,"gt_poses_t1.npy"), gt_t1)
    np.save(os.path.join(output_dir,"gt_poses_t2.npy"), gt_t2)

    # ---- VGGT 独立稠密推理 ----
    print("Running dense VGGT inference...")
    t0 = time.time()
    pred_t1 = run_vggt_inference(paths_t1, model)
    pred_t2 = run_vggt_inference(paths_t2, model)
    timing["dense_inference"] = time.time() - t0
    print(f"  Dense inference: {timing['dense_inference']:.4f}s")

    pred_poses_t1 = to_44(pred_t1["extrinsic"])
    pred_poses_t2 = to_44(pred_t2["extrinsic"])
    np.save(os.path.join(output_dir,"pred_poses_t1.npy"), pred_poses_t1)
    np.save(os.path.join(output_dir,"pred_poses_t2.npy"), pred_poses_t2)

    # ---- 点云提取 (适当降采样以加速纯几何ICP) ----
    print("Extracting points for Sim3 ICP...")
    pts1 = extract_high_conf_points(pred_t1, args.conf_threshold, max_points=50000)
    pts2 = extract_high_conf_points(pred_t2, args.conf_threshold, max_points=50000)

    # ---- Scale ICP (Sim3 ICP) 配准 ----
    print("Running custom Sim3 ICP (Scale ICP) registration...")
    fine_s, fine_R, fine_t, icp_info = scale_icp_registration(pts1, pts2)

    np.savez(os.path.join(output_dir,"fine_sim3.npz"),
             s=fine_s, R=fine_R, t=fine_t)

    # ---- 保存耗时与峰值显存 ----
    timing["paper_eval_time"] = icp_info["t_sim3_icp"]
    timing["total_pipeline"] = timing.get("dense_inference", 0.0) + timing["paper_eval_time"]
    timing["total"] = timing["total_pipeline"]

    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        timing["peak_memory_MB"] = peak_mem_mb
        print(f"  Peak GPU Memory: {peak_mem_mb:.2f} MB")
    else:
        timing["peak_memory_MB"] = 0.0

    with open(os.path.join(output_dir,"timing.json"),"w") as f:
        json.dump({**timing, **icp_info}, f, indent=2)

    print(f"  >> Paper Eval Time (Registration only): {timing['paper_eval_time']:.4f}s")
    print(f"  Results saved to {output_dir}")
    print(f"  Total pipeline time: {timing['total_pipeline']:.4f}s")

    del pred_t1, pred_t2, pts1, pts2
    torch.cuda.empty_cache(); gc.collect()

# ===========================================================================
# Entry
# ===========================================================================

def find_seq_dirs(images_dir):
    return sorted([d for d in os.listdir(images_dir)
                   if os.path.isdir(os.path.join(images_dir, d))])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      required=True)
    p.add_argument("--scene",          default=None)
    p.add_argument("--seq_t1",         default=None)
    p.add_argument("--seq_t2",         default=None)
    p.add_argument("--output_root",    default="./eval_results_sim3_icp")
    p.add_argument("--model_path",     default=os.environ.get("VGGT_MODEL_PATH", "./model.pt"))
    p.add_argument("--conf_threshold", type=int, default=50)
    return p.parse_args()

def main():
    args  = parse_args()
    model = load_model(args.model_path)

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted([d for d in os.listdir(args.data_root)
                         if os.path.isdir(os.path.join(args.data_root, d))])
        print(f"Found {len(scenes)} scenes: {scenes}")

    for scene in scenes:
        scene_dir  = os.path.join(args.data_root, scene)
        images_dir = os.path.join(scene_dir, "images")
        if not os.path.isdir(images_dir):
            print(f"[skip] {scene}: no images/"); continue

        seq_dirs = find_seq_dirs(images_dir)
        if len(seq_dirs) < 2:
            print(f"[skip] {scene}: need >=2 seq dirs, got {seq_dirs}"); continue

        seq_t1 = args.seq_t1 or seq_dirs[0]
        seq_t2 = args.seq_t2 or seq_dirs[1]
        print(f"\n{'='*60}")
        print(f"[Baseline Sim3 ICP]  Scene={scene}  T1={seq_t1}  T2={seq_t2}")
        print(f"{'='*60}")

        output_dir = os.path.join(args.output_root, scene)
        try:
            run_scene_sim3_icp(scene_dir, seq_t1, seq_t2, output_dir, model, args)
        except Exception as e:
            import traceback
            print(f"ERROR {scene}: {e}"); traceback.print_exc()

    print(f"\n{'='*60}")
    print("All baseline tasks completed successfully.")
    print("To evaluate the results, run the following command:")
    print(f"python eval_poses_1.py --eval_root {args.output_root}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()