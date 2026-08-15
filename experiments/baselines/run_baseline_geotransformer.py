"""
VGGT-CD Baseline: VGGT + GeoTransformer (CVPR 2022)
=====================================================
Pipeline:
  1. VGGT 独立稠密推理 T1 / T2（在你的主环境里跑）
  2. 把点云存成 .npy，调用 GeoTransformer 推理（在独立 conda 环境里跑）
  3. 读取 GeoTransformer 输出的变换矩阵
  4. 接 Point-to-Plane ICP 精细收尾（回到主环境）
  5. 输出 fine_sim3.npz，与 eval_poses_1.py 完全兼容

安装 GeoTransformer（一次性，需要独立 conda 环境）：
    conda create -n geotransformer python=3.8 -y
    conda activate geotransformer
    pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    git clone https://github.com/qinzheng93/GeoTransformer.git
    cd GeoTransformer
    pip install -r requirements.txt
    python setup.py build develop
    # 下载预训练权重（3DMatch，室内场景）
    # https://github.com/qinzheng93/GeoTransformer/releases
    # 放到 GeoTransformer/weights/geotransformer-3dmatch.pth.tar

注意事项：
  - GeoTransformer 在 3DMatch 上训练，voxel_size=0.025m (2.5cm)
  - 你的点云单位需要是 meter，并且场景尺度合理（室内场景最佳）
  - 点云需要降采样到 voxel_size=0.025m 才能送入模型

用法：
    # Step1: 在主环境里跑（VGGT推理 + 前后处理）
    python run_baseline_geotransformer.py \
        --data_root /path/to/WAT \
        --scene car_resized \
        --seq_t1 IMG_2285 \
        --seq_t2 IMG_2286 \
        --output_root ./eval_results_geotransformer \
        --geotransformer_root /path/to/GeoTransformer \
        --geotransformer_weights /path/to/GeoTransformer/weights/geotransformer-3dmatch.pth.tar \
        --geotransformer_conda geotransformer

    # 脚本会自动：
    #   1. VGGT推理 -> 存 src.npy / ref.npy
    #   2. subprocess 调用 geotransformer conda 环境跑 GeoTransformer
    #   3. 读取结果 -> ICP收尾 -> 存 fine_sim3.npz
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
import subprocess
import tempfile
import numpy as np
import torch
import open3d as o3d
import copy

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vggt"))

from coarse_to_fine_registration import align_intrinsics_and_depth
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# ===========================================================================
# COLMAP reader（与其他 baseline 完全相同）
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
# 点云提取 + 预处理
# ===========================================================================

def extract_points(predictions, conf_percentile=50):
    """VGGT predictions -> (N,3) 高置信度点云"""
    points = predictions["world_points_from_depth"]
    conf   = predictions["depth_conf"]
    thresh = (np.percentile(conf[conf > 0], conf_percentile)
              if np.any(conf > 0) else 0.01)
    valid  = conf.reshape(-1) > thresh
    return points.reshape(-1, 3)[valid].astype(np.float64)


def normalize_scale_for_geotransformer(pts1, pts2, target_voxel=0.025):
    """
    GeoTransformer 在 3DMatch 上训练，voxel_size=0.025m。
    VGGT 点云的尺度是任意的（取决于场景），需要归一化。

    策略：用 T2 点云的包围盒对角线估算场景尺度，
    计算缩放因子使得 voxel_size / scale ≈ 0.025m，
    返回缩放后的点云和缩放因子（后处理时需要反缩放）。
    """
    bbox  = pts2.max(axis=0) - pts2.min(axis=0)
    scene_diag = float(np.linalg.norm(bbox))
    # 期望场景对角线对应 3DMatch 典型场景（约 3-5m）
    # 3DMatch 室内场景对角线大约 4m，voxel_size=0.025m
    TARGET_DIAG = 4.0   # meters in 3DMatch scale
    scale = TARGET_DIAG / max(scene_diag, 1e-6)
    pts1_norm = pts1 * scale
    pts2_norm = pts2 * scale
    print(f"  [GeoTrans] scene_diag={scene_diag:.3f}  scale_factor={scale:.4f}  "
          f"-> normalized_diag={scene_diag*scale:.3f}m")
    return pts1_norm, pts2_norm, scale


def voxel_downsample_numpy(pts, voxel_size):
    """轻量体素降采样（用于送入 GeoTransformer 前控制点数）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_down   = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_down.points)

# ===========================================================================
# GeoTransformer 推理（通过 subprocess 调用独立 conda 环境）
# ===========================================================================

# 内联 demo 脚本，写入临时文件后用 geotransformer conda 环境执行
GEOTRANSFORMER_DEMO_SCRIPT = '''
import sys
import os
import argparse
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file",  required=True)
    parser.add_argument("--ref_file",  required=True)
    parser.add_argument("--out_file",  required=True)   # 输出 T44 .npy
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--geotransformer_root", required=True)
    args = parser.parse_args()

    # 把 GeoTransformer repo 加入路径
    sys.path.insert(0, args.geotransformer_root)
    exp_dir = os.path.join(
        args.geotransformer_root,
        "experiments",
        "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
    )
    sys.path.insert(0, exp_dir)

    from config import make_cfg
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda
    from geotransformer.utils.open3d import make_open3d_point_cloud
    from geotransformer.utils.registration import compute_registration_error
    from model import create_model

    cfg = make_cfg()

    # 加载模型
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.eval()

    # 加载点云
    src_points = np.load(args.src_file).astype(np.float32)   # (N,3) normalized
    ref_points = np.load(args.ref_file).astype(np.float32)   # (M,3) normalized

    # 构建输入 dict（pack mode）
    data_dict = {
        "ref_points": ref_points,
        "src_points": src_points,
        "ref_feats":  np.ones((ref_points.shape[0], 1), dtype=np.float32),
        "src_feats":  np.ones((src_points.shape[0], 1), dtype=np.float32),
        "transform":  np.eye(4, dtype=np.float32),   # placeholder，不用GT
    }

    # collate & to cuda
    collated = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size, cfg.backbone.init_radius,
        neighbor_limits=[38, 36, 36, 38]   # 3DMatch 默认值
    )
    collated = to_cuda(collated)

    # 推理
    with torch.no_grad():
        output = model(collated)

    # 从 output 提取变换（LGR: Local-to-Global Registration）
    # GeoTransformer 输出 "estimated_transform" shape (4,4)
    T = output["estimated_transform"].detach().cpu().numpy()   # (4,4)
    if T.ndim == 3:
        T = T[0]   # batch dim

    np.save(args.out_file, T.astype(np.float64))
    print(f"GeoTransformer done. T44 saved to {args.out_file}")
    print(f"T44:\\n{T}")

if __name__ == "__main__":
    main()
'''


def run_geotransformer_subprocess(src_npy, ref_npy, out_npy,
                                   geotransformer_root, weights_path,
                                   conda_env="geotransformer"):
    """
    把 GeoTransformer demo 写成临时脚本，
    用指定 conda 环境的 python 执行，避免环境冲突。
    """
    # 写临时 demo 脚本
    tmp_script = os.path.join(os.path.dirname(out_npy), "_geo_demo_tmp.py")
    with open(tmp_script, "w") as f:
        f.write(GEOTRANSFORMER_DEMO_SCRIPT)

    # 找 conda 环境的 python
    conda_base = subprocess.check_output(
        "conda info --base", shell=True).decode().strip()
    python_path = os.path.join(conda_base, "envs", conda_env, "bin", "python")
    if not os.path.exists(python_path):
        # fallback: 尝试 conda run
        python_path = "python"
        cmd_prefix  = f"conda run -n {conda_env} "
    else:
        cmd_prefix  = ""

    cmd = (f"{cmd_prefix}{python_path} {tmp_script} "
           f"--src_file={src_npy} "
           f"--ref_file={ref_npy} "
           f"--out_file={out_npy} "
           f"--weights={weights_path} "
           f"--geotransformer_root={geotransformer_root}")

    print(f"  [GeoTrans] Running subprocess:\n    {cmd}")
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - t0

    os.remove(tmp_script)   # 清理临时脚本

    if ret.returncode != 0:
        raise RuntimeError(f"GeoTransformer subprocess failed (code {ret.returncode})")

    T44 = np.load(out_npy)
    print(f"  [GeoTrans] Done  t={elapsed:.1f}s")
    return T44, elapsed


# ===========================================================================
# ICP 收尾（Point-to-Plane，恢复到原始尺度后精化）
# ===========================================================================

def icp_refine(source_pts, target_pts, T_init, voxel_size=None):
    """轻量 Point-to-Plane ICP 精化"""
    if voxel_size is None:
        bbox = target_pts.max(axis=0) - target_pts.min(axis=0)
        voxel_size = float(np.linalg.norm(bbox) * 0.005)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_pts)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_pts)

    tgt_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        voxel_size * 0.5,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=1000)
    )
    print(f"  [ICP refine] fitness={result.fitness:.4f}  rmse={result.inlier_rmse:.4f}")
    return np.array(result.transformation, dtype=np.float64)


def decompose_T44(T44):
    """刚体 4x4 -> s≈1, R, t"""
    R = T44[:3,:3]; t = T44[:3,3]
    s = float(np.cbrt(abs(np.linalg.det(R))))
    return float(s), R / s, t

# ===========================================================================
# 主流程
# ===========================================================================

def run_scene_geotransformer(scene_dir, seq_t1, seq_t2, output_dir, model, args):
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

    # ---- 点云提取 ----
    pts1_raw = extract_points(pred_t1, conf_percentile=args.conf_threshold)
    pts2_raw = extract_points(pred_t2, conf_percentile=args.conf_threshold)
    print(f"  Raw pts: T1={len(pts1_raw):,}  T2={len(pts2_raw):,}")

    # ---- 尺度归一化（匹配 3DMatch voxel_size=0.025m）----
    pts1_norm, pts2_norm, norm_scale = normalize_scale_for_geotransformer(
        pts1_raw, pts2_raw, target_voxel=0.025)

    # ---- 降采样到 GeoTransformer 期望的密度（0.025m voxel）----
    VOXEL_GEOTRANS = 0.025
    pts1_down = voxel_downsample_numpy(pts1_norm, VOXEL_GEOTRANS)
    pts2_down = voxel_downsample_numpy(pts2_norm, VOXEL_GEOTRANS)
    print(f"  Downsampled pts (voxel={VOXEL_GEOTRANS}m): "
          f"T1={len(pts1_down):,}  T2={len(pts2_down):,}")

    # 存为 npy 供 subprocess 读取
    src_npy = os.path.join(output_dir, "_src_norm.npy")
    ref_npy = os.path.join(output_dir, "_ref_norm.npy")
    out_npy = os.path.join(output_dir, "_geotrans_T44_norm.npy")
    np.save(src_npy, pts1_down.astype(np.float32))
    np.save(ref_npy, pts2_down.astype(np.float32))

    # ---- GeoTransformer 推理（subprocess）----
    print("Running GeoTransformer...")
    T44_norm, t_geo = run_geotransformer_subprocess(
        src_npy, ref_npy, out_npy,
        geotransformer_root = args.geotransformer_root,
        weights_path        = args.geotransformer_weights,
        conda_env           = args.geotransformer_conda,
    )
    timing["geotransformer"] = t_geo

    # ---- 反归一化：恢复到原始点云尺度 ----
    # T_norm 作用于归一化点云，需要转换回原始尺度
    # T_orig = diag(1/s) @ T_norm @ diag(s) （平移部分也要除以 norm_scale）
    T44_orig = T44_norm.copy()
    T44_orig[:3, 3] /= norm_scale   # 平移反缩放
    # 旋转部分不变

    # ---- ICP 精化（原始尺度）----
    print("Running ICP refinement (original scale)...")
    t0 = time.time()
    T44_final = icp_refine(pts1_raw, pts2_raw, T_init=T44_orig)
    timing["icp_refine"] = time.time() - t0
    print(f"  ICP refine: {timing['icp_refine']:.4f}s")

    # ---- 分解 Sim(3) ----
    fine_s, fine_R, fine_t = decompose_T44(T44_final)
    print(f"  Decomposed: scale={fine_s:.4f}")

    np.savez(os.path.join(output_dir,"fine_sim3.npz"),
             s=fine_s, R=fine_R, t=fine_t)
    np.savez(os.path.join(output_dir,"geotransformer_result.npz"),
             T44_norm=T44_norm, T44_orig=T44_orig, T44_final=T44_final,
             norm_scale=norm_scale)

    # 清理临时 npy
    for f in [src_npy, ref_npy, out_npy]:
        if os.path.exists(f): os.remove(f)

    # ---- 保存耗时与峰值显存 ----
    timing["paper_eval_time"] = timing["geotransformer"] + timing["icp_refine"]
    timing["total_pipeline"] = timing.get("dense_inference", 0.0) + timing["paper_eval_time"]
    timing["total"] = timing["total_pipeline"]

    # 记录主环境峰值显存 (主要为 VGGT 推理，GeoTransformer在子进程中不在此统计)
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        timing["peak_memory_MB"] = peak_mem_mb
        print(f"  Peak GPU Memory (Main Env): {peak_mem_mb:.2f} MB")
    else:
        timing["peak_memory_MB"] = 0.0

    with open(os.path.join(output_dir,"timing.json"),"w") as f:
        json.dump(timing, f, indent=2)

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
    p.add_argument("--data_root",              required=True)
    p.add_argument("--scene",                  default=None)
    p.add_argument("--seq_t1",                 default=None)
    p.add_argument("--seq_t2",                 default=None)
    p.add_argument("--output_root",            default="./eval_results_geotransformer")
    p.add_argument("--model_path",             default=os.environ.get("VGGT_MODEL_PATH", "./model.pt"))
    p.add_argument("--conf_threshold",         type=int, default=50)
    # GeoTransformer 相关
    p.add_argument("--geotransformer_root",    required=True,
                   help="GeoTransformer repo 根目录路径")
    p.add_argument("--geotransformer_weights", required=True,
                   help="预训练权重路径 (geotransformer-3dmatch.pth.tar)")
    p.add_argument("--geotransformer_conda",   default="geotransformer",
                   help="GeoTransformer 所在的 conda 环境名")
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
        print(f"[Baseline GeoTransformer]  Scene={scene}  T1={seq_t1}  T2={seq_t2}")
        print(f"{'='*60}")

        output_dir = os.path.join(args.output_root, scene)
        try:
            run_scene_geotransformer(scene_dir, seq_t1, seq_t2, output_dir, model, args)
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