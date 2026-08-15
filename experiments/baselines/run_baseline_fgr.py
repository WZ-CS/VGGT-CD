"""
VGGT-CD Baseline: VGGT + FGR + Colored ICP
============================================
Pipeline:
  1. VGGT 独立稠密推理 T1 / T2（与 RANSAC+ICP baseline 相同）
  2. FGR (Fast Global Registration, Zhou et al. ECCV 2016) 粗配准
     - 比 RANSAC 快 10-100x，不需要随机迭代
     - 用更小的 voxel_size (1%) 保留更多细节特征
  3. Colored ICP 精配准
     - 同时利用几何 + RGB 颜色约束（VGGT 输出点云有颜色）
     - 比 Point-to-Plane ICP 在有纹理场景更准
  4. 多分辨率 ICP 细化（粗→细 3级）进一步提升精度

输出格式与 run_inference.py / run_baseline_icp.py 完全一致。

用法：
    python run_baseline_fgr.py \
        --data_root /path/to/WAT \
        --scene car_resized \
        --seq_t1 IMG_2285 \
        --seq_t2 IMG_2286 \
        --output_root ./eval_results_fgr
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
    image_paths, gt_poses = [], []
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

# ===========================================================================
# 递归卸载工具
# ===========================================================================
def recursive_to_cpu(obj):
    """递归彻底将所有嵌套的张量转移至CPU，绝不留死角，防止GPU显存泄漏"""
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

    # 替换原本简单的 isinstance 检查，进行彻底的深度清理
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
# 点云构建
# ===========================================================================

def predictions_to_o3d_pcd(predictions, conf_percentile=50):
    """VGGT predictions -> open3d PointCloud（带颜色）"""
    points = predictions["world_points_from_depth"]   # (N,H,W,3)
    conf   = predictions["depth_conf"]                # (N,H,W)

    conf_thresh = (np.percentile(conf[conf > 0], conf_percentile)
                   if np.any(conf > 0) else 0.01)
    valid = conf.reshape(-1) > conf_thresh
    pts_flat = points.reshape(-1, 3)[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat.astype(np.float64))

    if "images" in predictions:
        imgs = predictions["images"]
        if imgs.shape[1] == 3:
            imgs = np.transpose(imgs, (0,2,3,1))
        colors = (np.clip(imgs, 0, 1) * 255).astype(np.uint8).reshape(-1,3)[valid]
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    return pcd

# ===========================================================================
# FGR + Colored ICP（多分辨率精化）
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


def fgr_colored_icp_registration(source_pcd, target_pcd, conf_percentile, voxel_size=None):
    """
    FGR 粗配准 + 多分辨率 Colored ICP 精配准

    精度优化策略：
      1. 小 voxel_size (1% scene scale) 保留更多局部特征
      2. FGR 代替 RANSAC，更快且对初始位姿更鲁棒
      3. 多分辨率 Colored ICP: 粗(1%) → 中(0.5%) → 细(0.2%)
         每级以上一级结果为初始值，逐步收敛
      4. 最后再跑一轮 Point-to-Plane ICP 作为几何收尾
    """
    # ---- 自适应 voxel_size：取场景对角线的 1%（比 ICP baseline 的 2% 更精细）----
    if voxel_size is None:
        bbox       = np.asarray(target_pcd.get_max_bound()) - \
                     np.asarray(target_pcd.get_min_bound())
        scene_diag = float(np.linalg.norm(bbox))
        voxel_size = scene_diag * 0.01
    print(f"  [FGR] voxel_size={voxel_size:.4f}")

    info = {"voxel_size": voxel_size}

    # ---- Step 1: FGR 粗配准 ----
    t0 = time.time()
    src_down, src_fpfh = preprocess_pcd(source_pcd, voxel_size)
    tgt_down, tgt_fpfh = preprocess_pcd(target_pcd, voxel_size)

    fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance = voxel_size * 1.5,
            iteration_number                = 128,      # 默认64，提高精度
            maximum_tuple_count             = 2000,     # 默认1000，更多tuple约束
        )
    )
    t_fgr = time.time() - t0
    info["t_fgr"]      = t_fgr
    info["fgr_fitness"] = fgr_result.fitness
    print(f"  [FGR] FGR done     t={t_fgr:.1f}s  fitness={fgr_result.fitness:.4f}")

    T_current = np.array(fgr_result.transformation, dtype=np.float64)

    # ---- Step 2: 多分辨率 Colored ICP ----
    # 确保点云有颜色（Colored ICP 需要）
    src_has_color = source_pcd.has_colors()
    tgt_has_color = target_pcd.has_colors()

    # 三级分辨率：粗→细
    multiscale_voxels = [voxel_size * 2.0,   # 粗：大范围拉入正确盆地
                         voxel_size * 1.0,   # 中：标准精度
                         voxel_size * 0.5]   # 细：精细收敛

    t0 = time.time()
    for level, vs in enumerate(multiscale_voxels):
        src_lvl = source_pcd.voxel_down_sample(vs)
        tgt_lvl = target_pcd.voxel_down_sample(vs)

        # 估计法线（Colored ICP 也需要法线）
        src_lvl.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vs*2, max_nn=30))
        tgt_lvl.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vs*2, max_nn=30))

        dist = vs * 0.4   # 收敛阈值随分辨率缩小

        if src_has_color and tgt_has_color:
            # Colored ICP: 几何 + 颜色联合优化
            result = o3d.pipelines.registration.registration_colored_icp(
                src_lvl, tgt_lvl,
                dist,
                T_current,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(
                    lambda_geometric=0.968   # 几何权重，原论文推荐值
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness  = 1e-6,
                    relative_rmse     = 1e-6,
                    max_iteration     = 100
                )
            )
        else:
            # 无颜色退化为 Point-to-Plane ICP
            result = o3d.pipelines.registration.registration_icp(
                src_lvl, tgt_lvl,
                dist, T_current,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )

        T_current = np.array(result.transformation, dtype=np.float64)
        print(f"  [FGR] Colored ICP lv{level}  voxel={vs:.4f}  "
              f"fitness={result.fitness:.4f}  rmse={result.inlier_rmse:.4f}")

    # ---- Step 3: 全分辨率 Point-to-Plane ICP 收尾 ----
    # 用原始点云（不降采样）做最终细化，最大化精度
    target_full = copy.deepcopy(target_pcd)
    target_full.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    final_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_full,
        voxel_size * 0.2,    # 非常小的收敛阈值
        T_current,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness = 1e-7,
            relative_rmse    = 1e-7,
            max_iteration    = 3000
        )
    )
    t_icp = time.time() - t0
    T_final = np.array(final_result.transformation, dtype=np.float64)

    info["t_colored_icp"]   = t_icp
    info["final_fitness"]   = final_result.fitness
    info["final_rmse"]      = final_result.inlier_rmse
    info["total_reg_time"]  = t_fgr + t_icp
    print(f"  [FGR] Final ICP done  t={t_icp:.1f}s  "
          f"fitness={final_result.fitness:.4f}  rmse={final_result.inlier_rmse:.4f}")

    return T_final, info


def decompose_sim3_from_T44(T44):
    """4x4 刚体变换 -> s, R, t（s 接近 1.0）"""
    R = T44[:3, :3]
    t = T44[:3,  3]
    s = float(np.cbrt(abs(np.linalg.det(R))))
    R_norm = R / s
    return float(s), R_norm, t

# ===========================================================================
# 主流程
# ===========================================================================

def run_scene_fgr(scene_dir, seq_t1, seq_t2, output_dir, model, args):
    os.makedirs(output_dir, exist_ok=True)
    timing = {}

    # 在场景开始前执行强制垃圾回收和显存清空
    if torch.cuda.is_available():
        import gc; gc.collect()
        torch.cuda.empty_cache()
        
        # 【诊断探针】记录重置前的基础显存
        base_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  [Debug] Base GPU Memory before scene: {base_mem:.2f} MB")
        
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

    # ---- VGGT 独立推理 ----
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
    print("Building point clouds...")
    src_pcd = predictions_to_o3d_pcd(pred_t1, conf_percentile=args.conf_threshold)
    tgt_pcd = predictions_to_o3d_pcd(pred_t2, conf_percentile=args.conf_threshold)
    print(f"  Source pts: {len(src_pcd.points):,}  Target pts: {len(tgt_pcd.points):,}")
    print(f"  Has color — src: {src_pcd.has_colors()}  tgt: {tgt_pcd.has_colors()}")

    # ---- FGR + Colored ICP ----
    print("Running FGR + Colored ICP registration...")
    t0 = time.time()
    T44, reg_info = fgr_colored_icp_registration(
        src_pcd, tgt_pcd,
        conf_percentile=args.conf_threshold
    )
    timing["fgr_colored_icp"] = time.time() - t0
    print(f"  Total registration: {timing['fgr_colored_icp']:.4f}s")

    # ---- 分解 Sim(3) ----
    fine_s, fine_R, fine_t = decompose_sim3_from_T44(T44)
    print(f"  Decomposed: scale={fine_s:.4f}")

    np.savez(os.path.join(output_dir,"fine_sim3.npz"),
             s=fine_s, R=fine_R, t=fine_t)
    np.savez(os.path.join(output_dir,"fgr_result.npz"),
             T44=T44,
             **{k: np.array(v) for k, v in reg_info.items()
                if isinstance(v, (int, float, np.ndarray))})

    # ---- 保存耗时与峰值显存 ----
    timing["paper_eval_time"] = timing.get("fgr_colored_icp", 0.0)
    timing["total_pipeline"] = timing.get("dense_inference", 0.0) + timing["paper_eval_time"]
    timing["total"] = timing["total_pipeline"]

    # 记录峰值显存
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
    p.add_argument("--output_root",    default="./eval_results_fgr")
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
            print(f"[skip] {scene}: need >=2 seq dirs"); continue

        seq_t1 = args.seq_t1 or seq_dirs[0]
        seq_t2 = args.seq_t2 or seq_dirs[1]
        print(f"\n{'='*60}")
        print(f"[Baseline FGR]  Scene={scene}  T1={seq_t1}  T2={seq_t2}")
        print(f"{'='*60}")

        output_dir = os.path.join(args.output_root, scene)
        try:
            run_scene_fgr(scene_dir, seq_t1, seq_t2, output_dir, model, args)
        except Exception as e:
            import traceback
            print(f"ERROR {scene}: {e}"); traceback.print_exc()

    # ---- 新增：推理结束后打印自动评估命令 ----
    print(f"\n{'='*60}")
    print("All baseline tasks completed successfully.")
    print("To evaluate the results, run the following command:")
    print(f"python eval_poses_1.py --eval_root {args.output_root}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()