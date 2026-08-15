"""
VGGT-CD Batch Inference Script
================================
数据结构：
    {data_root}/{scene}/
        images/
            {seq_t1}/   <- T1 图像序列（如 IMG_2285）
            {seq_t2}/   <- T2 图像序列（如 IMG_2286）
        sparse/0/
            cameras.bin
            images.bin
            points3D.bin

输出（每个场景）：
    {output_root}/{scene}/
        pred_poses_t1.npy   # VGGT预测位姿 (N, 4, 4)，world2cam，float64
        pred_poses_t2.npy
        gt_poses_t1.npy     # COLMAP GT位姿 (N, 4, 4)，world2cam，float64
        gt_poses_t2.npy
        image_names_t1.txt  # 对应图像文件名，逐行
        image_names_t2.txt
        coarse_sim3.npz     # Coarse Stage 输出的 Sim(3): s, R, t
        fine_sim3.npz       # Fine Stage 输出的 Sim(3): s, R, t
        timing.json         # 各阶段耗时

用法：
    python run_inference.py \
        --data_root /path/to/WAT \
        --scene car_resized \
        --seq_t1 IMG_2285 \
        --seq_t2 IMG_2286 \
        --output_root ./eval_results \
        --num_keyframes 5 \
        --conf_threshold 50
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import json
import time
import argparse
import numpy as np
import torch
import struct
import collections
import glob
from pathlib import Path

sys.path.append("vggt/")

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
# COLMAP binary reader
# ===========================================================================
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera      = collections.namedtuple("Camera",      ["id", "model", "width", "height", "params"])
Image       = collections.namedtuple("Image",       ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
}

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id, model_id, w, h = struct.unpack("<iiii", f.read(16))
            nparams = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
            cameras[cam_id] = Camera(cam_id, CAMERA_MODELS[model_id].model_name, w, h, params)
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec   = struct.unpack("<4d", f.read(32))   # (qw, qx, qy, qz)
            tvec   = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name   = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            name = name.decode()
            num_pts = struct.unpack("<Q", f.read(8))[0]
            xys         = struct.unpack(f"<{2*num_pts}d", f.read(16 * num_pts))
            point3D_ids = struct.unpack(f"<{num_pts}q",   f.read(8  * num_pts))
            images[img_id] = Image(img_id, qvec, tvec, cam_id, name,
                                   np.array(xys).reshape(-1, 2),
                                   np.array(point3D_ids))
    return images

def qvec2rotmat(qvec):
    """(qw, qx, qy, qz) -> 3x3 rotation matrix"""
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])

def colmap_image_to_pose44(img: Image) -> np.ndarray:
    """COLMAP Image -> 4x4 world2cam 变换矩阵"""
    R = qvec2rotmat(img.qvec)
    t = np.array(img.tvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

# ===========================================================================
# 读取 COLMAP GT 位姿，按图像文件名索引
# ===========================================================================
def load_colmap_poses(colmap_dir: str):
    """
    返回：{basename: 4x4 world2cam np.ndarray}
    basename = 图像文件名（不含路径），如 "DSC_0001.JPG"
    """
    images_bin = os.path.join(colmap_dir, "images.bin")
    colmap_images = read_images_binary(images_bin)
    pose_dict = {}
    for img in colmap_images.values():
        basename = os.path.basename(img.name)
        pose_dict[basename] = colmap_image_to_pose44(img)
    return pose_dict

# ===========================================================================
# 读取图像路径，匹配GT位姿
# ===========================================================================
def load_sequence(seq_dir: str, pose_dict: dict):
    """
    seq_dir: 包含图像的文件夹
    pose_dict: {basename: 4x4}
    返回：(image_paths, gt_poses) — 仅包含在 pose_dict 中存在的图像
    """
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_imgs = sorted([
        p for p in glob.glob(os.path.join(seq_dir, "*"))
        if os.path.splitext(p)[1] in exts
    ])

    image_paths = []
    gt_poses    = []
    for p in all_imgs:
        bn = os.path.basename(p)
        if bn in pose_dict:
            image_paths.append(p)
            gt_poses.append(pose_dict[bn])
        else:
            print(f"  [warn] no GT pose for {bn}, skipping")

    print(f"  Loaded {len(image_paths)} / {len(all_imgs)} images with GT poses from {seq_dir}")
    return image_paths, np.array(gt_poses, dtype=np.float64)  # (N, 4, 4)

# ===========================================================================
# VGGT 推理
# ===========================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path: str):
    print("Loading VGGT model...")
    m = VGGT()
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.eval()
    m = m.to(device)
    print("Model loaded.")
    return m

def run_vggt_inference(image_paths: list, model) -> dict:
    images = load_and_preprocess_images(image_paths).to(device)
    dtype  = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            arr = predictions[key].cpu().numpy()
            if arr.shape[0] == 1:
                arr = arr.squeeze(0)
            predictions[key] = arr

    intrinsic_ref = predictions["intrinsic"][0]
    intrinsic_aligned, depth_aligned = align_intrinsics_and_depth(
        predictions["intrinsic"], intrinsic_ref, predictions["depth"][..., 0]
    )
    predictions["intrinsic"] = intrinsic_aligned

    world_points = unproject_depth_map_to_point_map(
        depth_aligned[..., None], predictions["extrinsic"], intrinsic_aligned
    )
    predictions["world_points_from_depth"] = world_points

    del images
    torch.cuda.empty_cache()
    return predictions

# ===========================================================================
# Voxel-Hash One-Shot SVD Fine Alignment（复用app.py逻辑，去掉gradio依赖）
# ===========================================================================
def encode_voxel_keys(vox_coords, offset):
    return (vox_coords[:, 0] * offset * offset +
            vox_coords[:, 1] * offset +
            vox_coords[:, 2])

def one_shot_svd_fine_alignment(points1, conf1, points2, conf2,
                                 coarse_s, coarse_R, coarse_t,
                                 conf_percentile=50, max_points=30000):
    t0 = time.time()
    conf_thresh1 = np.percentile(conf1[conf1 > 0], conf_percentile) if np.any(conf1 > 0) else 0.01
    conf_thresh2 = np.percentile(conf2[conf2 > 0], conf_percentile) if np.any(conf2 > 0) else 0.01

    pts1_valid = points1.reshape(-1, 3)[conf1.reshape(-1) > conf_thresh1]
    pts2_valid = points2.reshape(-1, 3)[conf2.reshape(-1) > conf_thresh2]

    if len(pts1_valid) == 0 or len(pts2_valid) == 0:
        return coarse_s, coarse_R, coarse_t, {"fallback": True, "elapsed_time": time.time()-t0}

    pts1_aligned = apply_sim3(pts1_valid, coarse_s, coarse_R, coarse_t)

    bbox         = np.max(pts2_valid, axis=0) - np.min(pts2_valid, axis=0)
    scene_extent = float(np.percentile(bbox, 75))
    voxel_size   = float(np.clip(scene_extent * 0.005, 1e-4, 1.0))

    all_pts    = np.concatenate([pts1_aligned, pts2_valid], axis=0)
    global_min = all_pts.min(axis=0)
    vox1 = np.floor((pts1_aligned - global_min) / voxel_size).astype(np.int64)
    vox2 = np.floor((pts2_valid   - global_min) / voxel_size).astype(np.int64)
    offset = int(max(vox1.max(), vox2.max())) + 2

    keys1 = encode_voxel_keys(vox1, offset)
    keys2 = encode_voxel_keys(vox2, offset)
    _, idx1, idx2 = np.intersect1d(keys1, keys2, return_indices=True)

    if len(idx1) < 100:
        return coarse_s, coarse_R, coarse_t, {"fallback": True, "elapsed_time": time.time()-t0,
                                               "correspondences": len(idx1)}

    src = pts1_valid[idx1]
    tgt = pts2_valid[idx2]
    if len(src) > max_points:
        i = np.random.choice(len(src), max_points, replace=False)
        src, tgt = src[i], tgt[i]

    fine_s, fine_R, fine_t = umeyama_sim3(src, tgt, with_scale=True)
    elapsed = time.time() - t0
    print(f"  [Fine/SVD] s={fine_s:.4f}  corr={len(idx1):,}  t={elapsed:.2f}s")
    return fine_s, fine_R, fine_t, {"fallback": False, "elapsed_time": elapsed,
                                     "correspondences": len(idx1)}

# ===========================================================================
# 主推理函数
# ===========================================================================
def run_scene(scene_dir: str, seq_t1: str, seq_t2: str,
              output_dir: str, model, args):
    os.makedirs(output_dir, exist_ok=True)
    timing = {}

    # ---- 读取 COLMAP GT ----
    colmap_dir = os.path.join(scene_dir, "sparse", "0")
    print(f"Reading COLMAP GT from {colmap_dir}")
    pose_dict = load_colmap_poses(colmap_dir)

    seq_dir_t1 = os.path.join(scene_dir, "images", seq_t1)
    seq_dir_t2 = os.path.join(scene_dir, "images", seq_t2)
    paths_t1, gt_poses_t1 = load_sequence(seq_dir_t1, pose_dict)
    paths_t2, gt_poses_t2 = load_sequence(seq_dir_t2, pose_dict)

    if len(paths_t1) == 0 or len(paths_t2) == 0:
        print("ERROR: No images with GT poses found, skipping.")
        return

    # ---- 保存图像名 ----
    with open(os.path.join(output_dir, "image_names_t1.txt"), "w") as f:
        f.writelines(p + "\n" for p in paths_t1)
    with open(os.path.join(output_dir, "image_names_t2.txt"), "w") as f:
        f.writelines(p + "\n" for p in paths_t2)

    # ---- 保存GT位姿 ----
    np.save(os.path.join(output_dir, "gt_poses_t1.npy"), gt_poses_t1)
    np.save(os.path.join(output_dir, "gt_poses_t2.npy"), gt_poses_t2)
    print(f"  GT: T1={gt_poses_t1.shape}, T2={gt_poses_t2.shape}")

    # ---- Dense VGGT 推理 ----
    print("Running dense VGGT inference...")
    t0 = time.time()
    pred_t1 = run_vggt_inference(paths_t1, model)
    pred_t2 = run_vggt_inference(paths_t2, model)
    timing["dense_inference"] = time.time() - t0
    print(f"  Dense inference: {timing['dense_inference']:.1f}s")

    # extrinsic shape: (N, 3, 4) -> 补成 (N, 4, 4)
    def to_44(ext34):
        N = ext34.shape[0]
        T = np.zeros((N, 4, 4), dtype=np.float64)
        T[:, :3, :] = ext34.astype(np.float64)
        T[:,  3,  3] = 1.0
        return T

    pred_poses_t1 = to_44(pred_t1["extrinsic"])  # (N, 4, 4) world2cam
    pred_poses_t2 = to_44(pred_t2["extrinsic"])

    np.save(os.path.join(output_dir, "pred_poses_t1.npy"), pred_poses_t1)
    np.save(os.path.join(output_dir, "pred_poses_t2.npy"), pred_poses_t2)

    # ---- 置信度筛选 ----
    conf_pct = args.conf_threshold
    conf1 = pred_t1["depth_conf"].copy()
    conf2 = pred_t2["depth_conf"].copy()
    conf1[conf1 < np.percentile(conf1, conf_pct)] = 0.0
    conf2[conf2 < np.percentile(conf2, conf_pct)] = 0.0

    points1 = pred_t1["world_points_from_depth"]
    points2 = pred_t2["world_points_from_depth"]

    # ---- Coarse Stage ----
    print("Running Coarse Stage...")
    t0 = time.time()
    num_kf = min(args.num_keyframes, len(paths_t1), len(paths_t2))
    kf_t1  = select_keyframes(paths_t1, num_kf)
    kf_t2  = select_keyframes(paths_t2, num_kf)

    joint_paths    = [paths_t1[i] for i in kf_t1] + [paths_t2[i] for i in kf_t2]
    pred_joint     = run_vggt_inference(joint_paths, model)

    # D1 -> Joint
    pts_d1 = pred_t1["world_points_from_depth"][kf_t1].reshape(-1, 3)
    pts_j1 = pred_joint["world_points_from_depth"][:num_kf].reshape(-1, 3)
    c_d1   = pred_t1["depth_conf"][kf_t1].flatten()
    m1     = c_d1 > np.percentile(c_d1, conf_pct)
    p1, q1 = pts_d1[m1], pts_j1[m1]
    if len(p1) > 5000:
        i = np.random.choice(len(p1), 5000, replace=False); p1, q1 = p1[i], q1[i]
    s1, R1, t1 = umeyama_sim3(p1, q1, with_scale=True)

    # D2 -> Joint
    pts_d2 = pred_t2["world_points_from_depth"][kf_t2].reshape(-1, 3)
    pts_j2 = pred_joint["world_points_from_depth"][num_kf:].reshape(-1, 3)
    c_d2   = pred_t2["depth_conf"][kf_t2].flatten()
    m2     = c_d2 > np.percentile(c_d2, conf_pct)
    p2, q2 = pts_d2[m2], pts_j2[m2]
    if len(p2) > 5000:
        i = np.random.choice(len(p2), 5000, replace=False); p2, q2 = p2[i], q2[i]
    s2, R2, t2 = umeyama_sim3(p2, q2, with_scale=True)

    coarse_s = s1 / s2
    coarse_R = R2.T @ R1
    coarse_t = (1.0 / s2) * (R2.T @ (t1 - t2))
    timing["coarse"] = time.time() - t0
    print(f"  Coarse scale={coarse_s:.4f}  t={timing['coarse']:.1f}s")

    del pred_joint
    torch.cuda.empty_cache()

    np.savez(os.path.join(output_dir, "coarse_sim3.npz"),
             s=coarse_s, R=coarse_R, t=coarse_t)

    # ---- Fine Stage ----
    print("Running Fine Stage (Voxel-Hash SVD)...")
    t0 = time.time()
    fine_s, fine_R, fine_t, fine_info = one_shot_svd_fine_alignment(
        points1, conf1, points2, conf2,
        coarse_s, coarse_R, coarse_t,
        conf_percentile=conf_pct,
        max_points=30000,
    )
    timing["fine"] = time.time() - t0
    print(f"  Fine scale={fine_s:.4f}  t={timing['fine']:.1f}s")

    np.savez(os.path.join(output_dir, "fine_sim3.npz"),
             s=fine_s, R=fine_R, t=fine_t)

    # ---- 保存耗时 ----
    timing["total"] = sum(timing.values())
    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    print(f"  Total time: {timing['total']:.1f}s")
    print(f"  Results saved to {output_dir}")

    # ---- 清理 ----
    del pred_t1, pred_t2
    torch.cuda.empty_cache()
    import gc; gc.collect()


# ===========================================================================
# Entry Point
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",      type=str, required=True,
                        help="数据根目录，如 /path/to/WAT")
    parser.add_argument("--scene",          type=str, default=None,
                        help="单场景名，如 car_resized。不传则遍历 data_root 下所有子目录")
    parser.add_argument("--seq_t1",         type=str, default=None,
                        help="T1 图像文件夹名，如 IMG_2285。不传则自动取第一个子目录")
    parser.add_argument("--seq_t2",         type=str, default=None,
                        help="T2 图像文件夹名，如 IMG_2286。不传则自动取第二个子目录")
    parser.add_argument("--output_root",    type=str, default="./eval_results")
    parser.add_argument("--model_path",     type=str,
                        default=os.environ.get("VGGT_MODEL_PATH", "./model.pt"))
    parser.add_argument("--num_keyframes",  type=int, default=5)
    parser.add_argument("--conf_threshold", type=int, default=50,
                        help="置信度筛选百分位 (0-100)")
    return parser.parse_args()


def find_seq_dirs(images_dir: str):
    """返回 images/ 下排序后的子目录列表"""
    return sorted([
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ])


def main():
    args  = parse_args()
    model = load_model(args.model_path)

    # 确定要处理的场景列表
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted([
            d for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ])
        print(f"Found {len(scenes)} scenes: {scenes}")

    for scene in scenes:
        scene_dir = os.path.join(args.data_root, scene)
        images_dir = os.path.join(scene_dir, "images")

        if not os.path.isdir(images_dir):
            print(f"[skip] {scene}: no images/ dir")
            continue

        seq_dirs = find_seq_dirs(images_dir)
        if len(seq_dirs) < 2:
            print(f"[skip] {scene}: need at least 2 sequence dirs, found {seq_dirs}")
            continue

        seq_t1 = args.seq_t1 if args.seq_t1 else seq_dirs[0]
        seq_t2 = args.seq_t2 if args.seq_t2 else seq_dirs[1]
        print(f"\n{'='*60}")
        print(f"Scene: {scene}  T1={seq_t1}  T2={seq_t2}")
        print(f"{'='*60}")

        output_dir = os.path.join(args.output_root, scene)
        try:
            run_scene(scene_dir, seq_t1, seq_t2, output_dir, model, args)
        except Exception as e:
            import traceback
            print(f"ERROR in {scene}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()