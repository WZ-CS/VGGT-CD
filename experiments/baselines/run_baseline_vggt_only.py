"""
VGGT-CD Baseline: VGGT-Only (No Registration)
===============================================
最简baseline：T1/T2 分别独立用 VGGT 推理，不做任何跨时相配准。
直接把 T1 的坐标系当作 T2 坐标系（即 fine_sim3 = Identity）。

作用：
  - 证明 VGGT 独立推理存在尺度歧义和坐标系不一致问题
  - 作为"无配准上界"，衬托配准模块的必要性
  - ATE_Joint 应该会很大

输出格式与 run_inference.py / run_baseline_icp.py 完全一致，
直接用 eval_poses_1.py 评估。包含时间与显存占用统计。

用法：
    python run_baseline_vggt_only.py \
        --data_root /path/to/WAT \
        --scene car_resized \
        --seq_t1 IMG_2285 \
        --seq_t2 IMG_2286 \
        --output_root ./eval_results_vggt_only
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
        pose_dict[os.path.basename(img.name)] = _img_to_pose44(img)
    return pose_dict

def _img_to_pose44(img):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = qvec2rotmat(img.qvec)
    T[:3, 3] = np.array(img.tvec)
    return T

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
    T = np.zeros((N,4,4), dtype=np.float64)
    T[:,:3,:] = ext34.astype(np.float64)
    T[:,3,3]  = 1.0
    return T

# ===========================================================================
# 主流程
# ===========================================================================

def run_scene_vggt_only(scene_dir, seq_t1, seq_t2, output_dir, model, args):
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

    # ---- VGGT 独立推理，不做任何配准 ----
    print("Running VGGT-Only inference (no registration)...")
    t0 = time.time()
    pred_t1 = run_vggt_inference(paths_t1, model)
    pred_t2 = run_vggt_inference(paths_t2, model)
    timing["dense_inference"] = time.time() - t0
    print(f"  Dense inference: {timing['dense_inference']:.4f}s")

    pred_poses_t1 = to_44(pred_t1["extrinsic"])
    pred_poses_t2 = to_44(pred_t2["extrinsic"])
    np.save(os.path.join(output_dir,"pred_poses_t1.npy"), pred_poses_t1)
    np.save(os.path.join(output_dir,"pred_poses_t2.npy"), pred_poses_t2)

    # ---- fine_sim3 = Identity（无配准）----
    # s=1, R=I, t=0，即T1坐标系直接当T2坐标系用
    # eval_poses.py会用这个把T1变换到T2坐标系再评估Joint ATE
    # 尺度歧义会导致Joint ATE很大，这正是要展示的
    fine_s = 1.0
    fine_R = np.eye(3, dtype=np.float64)
    fine_t = np.zeros(3, dtype=np.float64)
    np.savez(os.path.join(output_dir,"fine_sim3.npz"),
             s=fine_s, R=fine_R, t=fine_t)

    # ---- 保存耗时与峰值显存 ----
    timing["registration"] = 0.0
    timing["paper_eval_time"] = 0.0  # Baseline中无配准，时间记作0.0
    timing["total_pipeline"] = timing.get("dense_inference", 0.0)
    timing["total"] = timing["total_pipeline"]

    # 记录峰值显存
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        timing["peak_memory_MB"] = peak_mem_mb
        print(f"  Peak GPU Memory: {peak_mem_mb:.2f} MB")
    else:
        timing["peak_memory_MB"] = 0.0

    with open(os.path.join(output_dir,"timing.json"),"w") as f:
        json.dump(timing, f, indent=2)

    print(f"  fine_sim3 = Identity (no registration)")
    print(f"  >> Paper Eval Time (Registration only): {timing['paper_eval_time']:.4f}s")
    print(f"  Results saved to {output_dir}")
    print(f"  Total pipeline time: {timing['total_pipeline']:.4f}s")

    del pred_t1, pred_t2
    torch.cuda.empty_cache(); gc.collect()


# ===========================================================================
# Entry
# ===========================================================================

def find_seq_dirs(images_dir):
    return sorted([d for d in os.listdir(images_dir)
                   if os.path.isdir(os.path.join(images_dir, d))])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--scene",       default=None)
    p.add_argument("--seq_t1",      default=None)
    p.add_argument("--seq_t2",      default=None)
    p.add_argument("--output_root", default="./eval_results_vggt_only")
    p.add_argument("--model_path",  default=os.environ.get("VGGT_MODEL_PATH", "./model.pt"))
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
        print(f"[VGGT-Only]  Scene={scene}  T1={seq_t1}  T2={seq_t2}")
        print(f"{'='*60}")

        output_dir = os.path.join(args.output_root, scene)
        try:
            run_scene_vggt_only(scene_dir, seq_t1, seq_t2, output_dir, model, args)
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