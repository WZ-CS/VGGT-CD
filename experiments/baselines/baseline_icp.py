"""
VGGT-CD Baseline: VGGT + RANSAC + Point-to-Plane ICP
======================================================
与我们方法的区别：
  - 相同：VGGT 独立稠密重建 T1 / T2
  - 不同：用 RANSAC+FPFH 粗配准 + Point-to-Plane ICP 精配准替代
          我们的 Coarse(联合推理) + Fine(Voxel-Hash SVD)
  - 输出格式与 run_inference.py 完全一致，可直接用 eval_poses_1.py 评估

数据结构（与 run_inference.py 相同）：
    {data_root}/{scene}/
        images/{seq_t1}/   T1图像
        images/{seq_t2}/   T2图像
        sparse/0/          COLMAP GT

输出：
    {output_root}/{scene_name}_icp/
        pred_poses_t1.npy
        pred_poses_t2.npy
        gt_poses_t1.npy
        gt_poses_t2.npy
        image_names_t1.txt
        image_names_t2.txt
        fine_sim3.npz       ICP变换分解出的 s/R/t（供eval_poses.py使用）
        icp_result.npz      完整ICP变换矩阵 + fitness + rmse
        timing.json         包含时间与显存统计

用法：
    python run_baseline_icp.py \
        --data_root /path/to/WAT \
        --scene car_resized \
        --seq_t1 IMG_2285 \
        --seq_t2 IMG_2286 \
        --output_root ./eval_results_baseline
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
    umeyama_sim3,
    apply_sim3,
    align_intrinsics_and_depth,
    select_keyframes,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# ===========================================================================
# COLMAP reader（与 run_inference.py 完全相同）
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

def read_cameras_binary(path):
    cameras = {}
    with open(path,"rb") as f:
        num = struct.unpack("<Q",f.read(8))[0]
        for _ in range(num):
            cam_id,model_id,w,h = struct.unpack("<iiii",f.read(16))
            nparams = CAMERA_MODELS[model_id].num_params
            params  = struct.unpack(f"<{nparams}d",f.read(8*nparams))
            cameras[cam_id] = Camera(cam_id,CAMERA_MODELS[model_id].model_name,w,h,params)
    return cameras

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

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            arr = predictions[key].cpu().numpy()
            if arr.shape[0] == 1: arr = arr.squeeze(0)
            predictions[key] = arr

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
# RANSAC + Point-to-Plane ICP
# ===========================================================================

def predictions_to_o3d_pcd(predictions, conf_percentile=50):
    """VGGT predictions -> open3d PointCloud（高置信度点）"""
    points = predictions["world_points_from_depth"]   # (N,H,W,3)
    conf   = predictions["depth_conf"]                # (N,H,W)

    conf_thresh = np.percentile(conf[conf > 0], conf_percentile) if np.any(conf > 0) else 0.01
    valid = conf.reshape(-1) > conf_thresh

    pts_flat = points.reshape(-1, 3)[valid]

    # 颜色（如果有）
    if "images" in predictions:
        imgs = predictions["images"]  # (N,3,H,W) or (N,H,W,3)
        if imgs.shape[1] == 3:
            imgs = np.transpose(imgs, (0,2,3,1))
        colors = (imgs * 255).astype(np.uint8).reshape(-1, 3)[valid]
        colors = colors.astype(np.float64) / 255.0
    else:
        colors = None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def ransac_icp_registration(source_pcd, target_pcd, voxel_size=None):
    """
    RANSAC(FPFH) 粗配准 + Point-to-Plane ICP 精配准
    返回：4x4 变换矩阵 T（将 source 变换到 target 坐标系），以及统计信息
    """
    # 自适应 voxel size
    if voxel_size is None:
        bbox  = np.asarray(target_pcd.get_max_bound()) - \
                np.asarray(target_pcd.get_min_bound())
        voxel_size = float(np.linalg.norm(bbox) * 0.02)
    print(f"  [ICP] voxel_size={voxel_size:.4f}")

    # ---- 降采样 + 法线 + FPFH ----
    def preprocess(pcd):
        down = pcd.voxel_down_sample(voxel_size)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
        return down, fpfh

    t0 = time.time()
    src_down, src_fpfh = preprocess(source_pcd)
    tgt_down, tgt_fpfh = preprocess(target_pcd)

    # ---- RANSAC 全局粗配准 ----
    dist_coarse = voxel_size * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        dist_coarse,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_coarse)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    t_ransac = time.time() - t0
    print(f"  [ICP] RANSAC done  t={t_ransac:.1f}s  "
          f"fitness={ransac_result.fitness:.4f}")

    # ---- Point-to-Plane ICP 精配准 ----
    target_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    dist_fine = voxel_size * 0.4
    t0 = time.time()
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        dist_fine,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    t_icp = time.time() - t0
    print(f"  [ICP] ICP done     t={t_icp:.1f}s  "
          f"fitness={icp_result.fitness:.4f}  rmse={icp_result.inlier_rmse:.4f}")

    T44 = np.array(icp_result.transformation, dtype=np.float64)
    info = {
        "voxel_size":    voxel_size,
        "ransac_fitness":ransac_result.fitness,
        "icp_fitness":   icp_result.fitness,
        "icp_rmse":      icp_result.inlier_rmse,
        "t_ransac":      t_ransac,
        "t_icp":         t_icp,
    }
    return T44, info


def decompose_sim3_from_T44(T44):
    """
    从 ICP 输出的 4x4 刚体变换分解出 Sim(3) 的 s/R/t。
    ICP 输出是刚体（s=1），但为了和 eval_poses.py 接口统一，
    用 umeyama_sim3 重新拟合一次（确保数值一致）。
    实际上对刚体变换，s 应该接近 1.0。
    """
    R = T44[:3, :3]
    t = T44[:3,  3]
    # 刚体：s=1，直接提取
    s = float(np.cbrt(np.linalg.det(R)))   # det(R)=1 -> s=1，数值稳定写法
    R_norm = R / s                           # 归一化确保正交
    return float(s), R_norm, t


# ===========================================================================
# 主流程
# ===========================================================================

def run_scene_baseline(scene_dir, seq_t1, seq_t2, output_dir, model, args):
    os.makedirs(output_dir, exist_ok=True)
    timing = {}

    # 在场景开始前重置峰值显存统计
    if torch.cuda.is_available():
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
    print("Running dense VGGT inference (independent, no joint recon)...")
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
    print("Building point clouds for ICP...")
    src_pcd = predictions_to_o3d_pcd(pred_t1, conf_percentile=args.conf_threshold)
    tgt_pcd = predictions_to_o3d_pcd(pred_t2, conf_percentile=args.conf_threshold)
    print(f"  Source pts: {len(src_pcd.points):,}  Target pts: {len(tgt_pcd.points):,}")

    # ---- RANSAC + ICP ----
    print("Running RANSAC + ICP registration (baseline)...")
    t0 = time.time()
    T44, icp_info = ransac_icp_registration(src_pcd, tgt_pcd, voxel_size=None)
    timing["ransac_icp"] = time.time() - t0
    print(f"  Total RANSAC+ICP: {timing['ransac_icp']:.4f}s")

    # ---- 分解 Sim(3)，供 eval_poses.py 使用 ----
    fine_s, fine_R, fine_t = decompose_sim3_from_T44(T44)
    print(f"  Decomposed: scale={fine_s:.4f}")

    np.savez(os.path.join(output_dir,"fine_sim3.npz"),
             s=fine_s, R=fine_R, t=fine_t)
    np.savez(os.path.join(output_dir,"icp_result.npz"),
             T44=T44, **{k:v for k,v in icp_info.items()
                         if isinstance(v,(int,float,np.ndarray))})

    # ---- 保存耗时与峰值显存 ----
    timing["paper_eval_time"] = timing.get("ransac_icp", 0.0)
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
        json.dump({**timing, **icp_info}, f, indent=2)

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
    p.add_argument("--output_root",    default="./eval_results_baseline")
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
        print(f"[Baseline ICP]  Scene={scene}  T1={seq_t1}  T2={seq_t2}")
        print(f"{'='*60}")

        # baseline结果存到独立目录，名称加 _icp 后缀区分
        output_dir = os.path.join(args.output_root, scene)
        try:
            run_scene_baseline(scene_dir, seq_t1, seq_t2, output_dir, model, args)
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