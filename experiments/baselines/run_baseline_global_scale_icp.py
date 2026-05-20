"""
VGGT-CD Baseline: Global Scale Prior + Standard SE(3) ICP
===========================================================
两步法（解耦法）Baseline：
  1. VGGT 独立稠密推理 T1 / T2
  2. 提取全局尺度先验（Global Scale Prior）：
     利用 T1 和 T2 点云包围盒（Bounding Box）的对角线长度比例，估算全局缩放尺度 s。
  3. 将尺度 s 直接乘到 T1 点云上，使其与 T2 尺度一致。
  4. 使用标准 SE(3) 刚体配准（RANSAC + Point-to-Plane ICP）计算 R 和 t。
  
核心证明目的：
  证明基于全局几何特征（如包围盒）计算尺度的方法极度脆弱。
  在跨时相变化检测中，真实的物理变化（如新增的车辆）会改变包围盒大小，
  导致提取的尺度先验 s 错误，进而使得后续的 ICP 无论迭代多少次都无法对齐静态背景。

输出格式与 run_inference.py 完全一致，可直接用 eval_poses_1.py 评估。

用法：
    python run_baseline_global_scale_icp.py \
        --data_root /path/to/WAT \
        --scene car_resized \
        --seq_t1 IMG_2285 \
        --seq_t2 IMG_2286 \
        --output_root ./eval_results_global_scale_icp
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
import open3d as o3d
import copy

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vggt"))

from coarse_to_fine_registration import (
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
Image       = collections.namedtuple("Image",["id","qvec","tvec","camera_id","name","xys","point3D_ids"])

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
    return np.array([[1-2*(qy**2+qz**2),  2*(qx*qy-qw*qz),  2*(qx*qz+qw*qy)],[  2*(qx*qy+qw*qz),1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],[  2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx),1-2*(qx**2+qy**2)],
    ])

def load_colmap_poses(colmap_dir):
    pose_dict = {}
    for img in read_images_binary(os.path.join(colmap_dir,"images.bin")).values():
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = qvec2rotmat(img.qvec)
        T[:3, 3] = np.array(img.tvec)
        pose_dict[os.path.basename(img.name)] = T
    return pose_dict

def load_sequence(seq_dir, pose_dict):
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    all_imgs = sorted([p for p in glob.glob(os.path.join(seq_dir,"*"))
                       if os.path.splitext(p)[1] in exts])
    image_paths, gt_poses = [],[]
    for p in all_imgs:
        bn = os.path.basename(p)
        if bn in pose_dict:
            image_paths.append(p); gt_poses.append(pose_dict[bn])
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
    T = np.zeros((N,4,4), dtype=np.float64)
    T[:,:3,:] = ext34.astype(np.float64)
    T[:,3,3]  = 1.0
    return T


# ===========================================================================
# 点云处理
# ===========================================================================
def predictions_to_o3d_pcd(predictions, conf_percentile=50):
    points = predictions["world_points_from_depth"]
    conf   = predictions["depth_conf"]
    thresh = np.percentile(conf[conf > 0], conf_percentile) if np.any(conf > 0) else 0.01
    valid  = conf.reshape(-1) > thresh
    pts_flat = points.reshape(-1, 3)[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat.astype(np.float64))
    return pcd


# ===========================================================================
# Global Scale + Standard SE(3) ICP 配准逻辑
# ===========================================================================
def preprocess_pcd(pcd, voxel_size):
    """降采样 + 法线 + FPFH"""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return down, fpfh


def global_scale_icp_registration(source_pcd, target_pcd, voxel_size=None):
    t0 = time.time()
    
    # --- 1. 计算全局尺度先验 (Global Scale Prior) ---
    bbox_src = np.asarray(source_pcd.get_max_bound()) - np.asarray(source_pcd.get_min_bound())
    bbox_tgt = np.asarray(target_pcd.get_max_bound()) - np.asarray(target_pcd.get_min_bound())
    diag_src = np.linalg.norm(bbox_src)
    diag_tgt = np.linalg.norm(bbox_tgt)
    
    global_scale = float(diag_tgt / max(diag_src, 1e-6))
    print(f"  [Global-Scale] Source diag: {diag_src:.3f}, Target diag: {diag_tgt:.3f}")
    print(f"  [Global-Scale] Estimated Global Scale: {global_scale:.4f}")

    # --- 2. 尺度对齐：缩放源点云 ---
    source_pcd_scaled = copy.deepcopy(source_pcd)
    # 以原点为中心进行缩放，确保只受纯数学尺度影响，避免平移耦合
    source_pcd_scaled.scale(global_scale, center=np.zeros(3))

    if voxel_size is None:
        voxel_size = float(diag_tgt * 0.02)
    print(f"  [ICP] voxel_size={voxel_size:.4f}")

    # --- 3. 粗配准 (RANSAC 刚体对齐) ---
    src_down, src_fpfh = preprocess_pcd(source_pcd_scaled, voxel_size)
    tgt_down, tgt_fpfh = preprocess_pcd(target_pcd, voxel_size)
    
    dist_coarse = voxel_size * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        dist_coarse,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_coarse)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    t_ransac = time.time() - t0
    print(f"  [ICP] RANSAC done  t={t_ransac:.1f}s  fitness={ransac_result.fitness:.4f}")

    # --- 4. 精配准 (Point-to-Plane ICP 刚体对齐) ---
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    dist_fine = voxel_size * 0.4
    t1 = time.time()
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd_scaled, target_pcd,
        dist_fine,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
    )
    t_icp = time.time() - t1
    print(f"  [ICP] SE(3) ICP done t={t_icp:.1f}s  fitness={icp_result.fitness:.4f}  rmse={icp_result.inlier_rmse:.4f}")

    # --- 5. 组装最终的 Sim(3) 结果 ---
    # ICP 的变换矩阵是对缩放后的点云做 SE(3) (即 R*(s*P) + t)
    # 所以 s = global_scale, R = icp_R, t = icp_t
    T_icp = np.array(icp_result.transformation, dtype=np.float64)
    R_final = T_icp[:3, :3]
    t_final = T_icp[:3, 3]
    s_final = global_scale

    info = {
        "global_scale": global_scale,
        "ransac_fitness": ransac_result.fitness,
        "icp_fitness": icp_result.fitness,
        "icp_rmse": icp_result.inlier_rmse,
        "t_ransac": t_ransac,
        "t_icp": t_icp,
        "t_total_reg": (time.time() - t0)
    }
    
    return s_final, R_final, t_final, info


# ===========================================================================
# 主流程
# ===========================================================================
def run_scene_global_scale_icp(scene_dir, seq_t1, seq_t2, output_dir, model, args):
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

    # ---- VGGT 推理 ----
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

    # ---- 点云构建 ----
    print("Building point clouds for Global Scale + ICP...")
    src_pcd = predictions_to_o3d_pcd(pred_t1, conf_percentile=args.conf_threshold)
    tgt_pcd = predictions_to_o3d_pcd(pred_t2, conf_percentile=args.conf_threshold)

    # ---- Global Scale + ICP ----
    print("Running Global Scale estimation + SE(3) ICP registration...")
    fine_s, fine_R, fine_t, reg_info = global_scale_icp_registration(src_pcd, tgt_pcd)

    np.savez(os.path.join(output_dir,"fine_sim3.npz"),
             s=fine_s, R=fine_R, t=fine_t)

    # ---- 保存耗时与峰值显存 ----
    timing["paper_eval_time"] = reg_info["t_total_reg"]
    timing["total_pipeline"] = timing.get("dense_inference", 0.0) + timing["paper_eval_time"]
    timing["total"] = timing["total_pipeline"]

    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        timing["peak_memory_MB"] = peak_mem_mb
        print(f"  Peak GPU Memory: {peak_mem_mb:.2f} MB")
    else:
        timing["peak_memory_MB"] = 0.0

    with open(os.path.join(output_dir,"timing.json"),"w") as f:
        json.dump({**timing, **reg_info}, f, indent=2)

    print(f"  >> Paper Eval Time (Registration only): {timing['paper_eval_time']:.4f}s")
    print(f"  Results saved to {output_dir}")
    print(f"  Total pipeline time: {timing['total_pipeline']:.4f}s")

    del pred_t1, pred_t2, src_pcd, tgt_pcd
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
    p.add_argument("--output_root",    default="./eval_results_global_scale_icp")
    p.add_argument("--model_path",     default=os.environ.get("VGGT_MODEL_PATH", "./model.pt"))
    p.add_argument("--conf_threshold", type=int, default=50)
    return p.parse_args()

def main():
    args  = parse_args()
    model = load_model(args.model_path)

    if args.scene:
        scenes =[args.scene]
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
            print(f"[skip] {scene}: need >=2 seq dirs"); continue

        seq_t1 = args.seq_t1 or seq_dirs[0]
        seq_t2 = args.seq_t2 or seq_dirs[1]
        print(f"\n{'='*60}")
        print(f"[Baseline Global-Scale ICP]  Scene={scene}  T1={seq_t1}  T2={seq_t2}")
        print(f"{'='*60}")

        output_dir = os.path.join(args.output_root, scene)
        try:
            run_scene_global_scale_icp(scene_dir, seq_t1, seq_t2, output_dir, model, args)
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
