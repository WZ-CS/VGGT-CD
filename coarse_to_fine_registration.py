# Copyright (c) 2024
# VGGT-CD: Coarse-to-Fine Bi-temporal Point Cloud Registration
# Two-stage registration: Joint reconstruction for coarse alignment + Static region refinement

import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict, List
import time


def umeyama_sim3(X: np.ndarray, Y: np.ndarray, with_scale: bool = True,
                  weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Sim(3) transformation using Umeyama algorithm.
    Finds s, R, t such that Y ≈ s * R @ X + t
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert X.shape == Y.shape and X.shape[1] == 3
    N = X.shape[0]

    if weights is None:
        w = np.ones(N)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        assert w.shape[0] == N
        w = np.clip(w, 1e-12, None)
    w = w / np.sum(w)

    mu_x = np.sum(X * w[:, None], axis=0)
    mu_y = np.sum(Y * w[:, None], axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    S = (Xc * w[:, None]).T @ Yc
    U, D, Vt = np.linalg.svd(S)
    V = Vt.T

    sgn = np.sign(np.linalg.det(V @ U.T))
    C = np.eye(3)
    C[-1, -1] = sgn
    R = V @ C @ U.T

    if with_scale:
        var_x = np.sum(w * np.sum(Xc**2, axis=1))
        c = np.ones(3)
        c[-1] = sgn
        s = (D * c).sum() / (var_x + 1e-12)
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)
    return s, R, t


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply Sim(3) transformation to points."""
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    transformed = s * (points_flat @ R.T) + t
    return transformed.reshape(original_shape)


def select_keyframes(image_paths: List[str], num_keyframes: int = 5) -> List[int]:
    """
    Select keyframes uniformly from image sequence.
    """
    n = len(image_paths)
    if n <= num_keyframes:
        return list(range(n))

    # Uniform selection
    indices = np.linspace(0, n - 1, num_keyframes, dtype=int)
    return indices.tolist()


def align_intrinsics_and_depth(Ks: np.ndarray, K_ref: np.ndarray, depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align all intrinsics to reference K and rescale depths accordingly.
    From SwiftVGGT: ensures consistent scale across frames.
    """
    N, H, W = depths.shape
    depths_aligned = np.zeros_like(depths)
    Ks_aligned = np.zeros_like(Ks)

    for i in range(N):
        Ki = Ks[i]
        scale_x = K_ref[0, 0] / Ki[0, 0]
        scale_y = K_ref[1, 1] / Ki[1, 1]
        scale_mean = 0.5 * (scale_x + scale_y)

        K_new = Ki.copy()
        K_new[0, 0] = K_ref[0, 0]
        K_new[1, 1] = K_ref[1, 1]
        K_new[0, 2] = K_ref[0, 2]
        K_new[1, 2] = K_ref[1, 2]

        Ks_aligned[i] = K_new
        depths_aligned[i] = depths[i] * scale_mean

    return Ks_aligned, depths_aligned


class CoarseToFineRegistration:
    """
    Two-stage bi-temporal point cloud registration.

    Stage 1 (Coarse): Joint reconstruction of keyframes from T1 and T2
                      to establish initial coordinate alignment
    Stage 2 (Fine): Static region detection and iterative refinement
    """

    def __init__(self,
                 num_keyframes: int = 5,
                 depth_diff_threshold: float = 0.1,
                 conf_threshold: float = 0.5,
                 max_refine_iterations: int = 5):
        """
        Args:
            num_keyframes: Number of keyframes to use for coarse alignment
            depth_diff_threshold: Threshold for depth consistency check
            conf_threshold: Confidence threshold multiplier
            max_refine_iterations: Maximum refinement iterations
        """
        self.num_keyframes = num_keyframes
        self.depth_diff_threshold = depth_diff_threshold
        self.conf_threshold = conf_threshold
        self.max_refine_iterations = max_refine_iterations

    def compute_coarse_alignment(self,
                                  joint_predictions: Dict,
                                  n_t1: int,
                                  n_t2: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Stage 1: Extract coarse Sim(3) from joint reconstruction.

        The joint reconstruction already puts T1 and T2 points in the same
        coordinate frame. We extract the relative transformation by comparing
        the reconstructed poses.

        Args:
            joint_predictions: VGGT predictions from joint T1+T2 reconstruction
            n_t1: Number of T1 keyframes
            n_t2: Number of T2 keyframes

        Returns:
            s, R, t: Coarse Sim(3) transformation from T1 to T2 coordinate system
        """
        # In joint reconstruction, all frames share the same coordinate system
        # The "transformation" is already applied - we just need to verify consistency

        # Get world points
        world_points = joint_predictions['world_points_from_depth']
        depth_conf = joint_predictions['depth_conf']

        # Split into T1 and T2
        points_t1_joint = world_points[:n_t1]
        points_t2_joint = world_points[n_t1:n_t1+n_t2]
        conf_t1_joint = depth_conf[:n_t1]
        conf_t2_joint = depth_conf[n_t1:n_t1+n_t2]

        # For coarse alignment, identity transform since joint reconstruction
        # already aligns everything
        # But we return the centroid offset for proper alignment

        # Flatten and filter by confidence
        pts1_flat = points_t1_joint.reshape(-1, 3)
        pts2_flat = points_t2_joint.reshape(-1, 3)
        conf1_flat = conf_t1_joint.flatten()
        conf2_flat = conf_t2_joint.flatten()

        # High confidence points
        mask1 = conf1_flat > np.percentile(conf1_flat, 50)
        mask2 = conf2_flat > np.percentile(conf2_flat, 50)

        pts1_high = pts1_flat[mask1]
        pts2_high = pts2_flat[mask2]

        # Compute centroids
        centroid1 = np.mean(pts1_high, axis=0)
        centroid2 = np.mean(pts2_high, axis=0)

        print(f"[Coarse] T1 centroid: {centroid1}")
        print(f"[Coarse] T2 centroid: {centroid2}")
        print(f"[Coarse] Centroid offset: {np.linalg.norm(centroid2 - centroid1):.4f}")

        # Joint reconstruction gives identity transform (points already aligned)
        return 1.0, np.eye(3), np.zeros(3)

    def find_static_correspondences(self,
                                     points1: np.ndarray,
                                     conf1: np.ndarray,
                                     points2: np.ndarray,
                                     conf2: np.ndarray,
                                     depth1: np.ndarray,
                                     depth2: np.ndarray,
                                     current_s: float,
                                     current_R: np.ndarray,
                                     current_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find static region correspondences using depth consistency.

        Following SwiftVGGT's approach: use depth difference and confidence
        to identify static (unchanged) regions.

        Args:
            points1, conf1, depth1: T1 dense reconstruction
            points2, conf2, depth2: T2 dense reconstruction
            current_s, R, t: Current Sim(3) estimate

        Returns:
            src_points, tgt_points, weights: Corresponding static points
        """
        # Apply current transformation to T1 points
        points1_transformed = apply_sim3(points1, current_s, current_R, current_t)

        # Flatten
        pts1_flat = points1_transformed.reshape(-1, 3)
        pts2_flat = points2.reshape(-1, 3)
        conf1_flat = conf1.flatten()
        conf2_flat = conf2.flatten()
        depth1_flat = depth1.flatten()
        depth2_flat = depth2.flatten()

        # Build KD-tree for T2
        tree = cKDTree(pts2_flat)

        # Find nearest neighbors
        distances, indices = tree.query(pts1_flat, k=1)

        # Confidence thresholds
        conf1_thresh = np.mean(conf1_flat) * self.conf_threshold
        conf2_thresh = np.mean(conf2_flat) * self.conf_threshold

        # Depth difference (in relative terms)
        depth1_matched = depth1_flat
        depth2_matched = depth2_flat[indices]

        # Normalize depth difference by mean depth
        mean_depth = (np.mean(depth1_matched) + np.mean(depth2_matched)) / 2
        rel_depth_diff = np.abs(depth1_matched - depth2_matched) / (mean_depth + 1e-6)

        # Static region mask:
        # 1. High confidence in both
        # 2. Similar depth values (indicating same surface)
        # 3. Small correspondence distance
        median_dist = np.median(distances)

        static_mask = (
            (conf1_flat > conf1_thresh) &
            (conf2_flat[indices] > conf2_thresh) &
            (rel_depth_diff < self.depth_diff_threshold) &
            (distances < median_dist * 2)
        )

        num_static = np.sum(static_mask)
        print(f"[Static] Found {num_static} static correspondences "
              f"({100*num_static/len(static_mask):.1f}%)")

        if num_static < 100:
            # Relax threshold
            static_mask = (
                (conf1_flat > conf1_thresh * 0.5) &
                (conf2_flat[indices] > conf2_thresh * 0.5) &
                (distances < median_dist * 3)
            )
            num_static = np.sum(static_mask)
            print(f"[Static] Relaxed: {num_static} correspondences")

        # Get original (untransformed) source points
        pts1_orig_flat = points1.reshape(-1, 3)

        src_points = pts1_orig_flat[static_mask]
        tgt_points = pts2_flat[indices[static_mask]]

        # Weights based on confidence product
        weights = conf1_flat[static_mask] * conf2_flat[indices[static_mask]]
        weights = weights / np.sum(weights)

        return src_points, tgt_points, weights

    def refine_alignment(self,
                         points1: np.ndarray,
                         conf1: np.ndarray,
                         depth1: np.ndarray,
                         points2: np.ndarray,
                         conf2: np.ndarray,
                         depth2: np.ndarray,
                         init_s: float,
                         init_R: np.ndarray,
                         init_t: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, Dict]:
        """
        Stage 2: Iterative refinement using static regions.

        Args:
            points1, conf1, depth1: T1 dense reconstruction
            points2, conf2, depth2: T2 dense reconstruction
            init_s, R, t: Initial Sim(3) from coarse alignment

        Returns:
            s, R, t: Refined Sim(3) transformation
            info: Dictionary with additional information
        """
        s, R, t = init_s, init_R, init_t

        errors = []

        for iteration in range(self.max_refine_iterations):
            # Find static correspondences
            src_pts, tgt_pts, weights = self.find_static_correspondences(
                points1, conf1, points2, conf2, depth1, depth2, s, R, t
            )

            if len(src_pts) < 50:
                print(f"[Refine] Iteration {iteration}: Too few correspondences, stopping")
                break

            # Compute new Sim(3) using weighted Umeyama
            s_new, R_new, t_new = umeyama_sim3(src_pts, tgt_pts, with_scale=True, weights=weights)

            # Compute error
            transformed = apply_sim3(src_pts, s_new, R_new, t_new)
            error = np.mean(np.linalg.norm(transformed - tgt_pts, axis=1))
            errors.append(error)

            print(f"[Refine] Iteration {iteration}: scale={s_new:.4f}, error={error:.6f}")

            # Check convergence
            if iteration > 0:
                rel_change = abs(errors[-1] - errors[-2]) / (errors[-2] + 1e-8)
                if rel_change < 0.01:
                    print(f"[Refine] Converged at iteration {iteration}")
                    break

            s, R, t = s_new, R_new, t_new

        info = {
            'num_iterations': len(errors),
            'final_error': errors[-1] if errors else float('inf'),
            'error_history': errors,
            'num_correspondences': len(src_pts) if 'src_pts' in dir() else 0
        }

        return s, R, t, info

    def detect_changes(self,
                       points1: np.ndarray,
                       conf1: np.ndarray,
                       colors1: np.ndarray,
                       points2: np.ndarray,
                       conf2: np.ndarray,
                       colors2: np.ndarray,
                       s: float,
                       R: np.ndarray,
                       t: np.ndarray,
                       change_threshold: float = 0.2) -> Dict:
        """
        Detect changes after alignment.

        Args:
            points1, conf1, colors1: T1 data
            points2, conf2, colors2: T2 data
            s, R, t: Final Sim(3) transformation
            change_threshold: Threshold for change detection

        Returns:
            Dictionary with change detection results
        """
        # Transform T1 points
        points1_aligned = apply_sim3(points1, s, R, t)

        # Flatten
        pts1_flat = points1_aligned.reshape(-1, 3)
        pts2_flat = points2.reshape(-1, 3)
        conf1_flat = conf1.flatten()
        conf2_flat = conf2.flatten()
        col1_flat = colors1.reshape(-1, 3)
        col2_flat = colors2.reshape(-1, 3)

        # Filter by confidence
        conf1_thresh = np.percentile(conf1_flat, 30)
        conf2_thresh = np.percentile(conf2_flat, 30)

        mask1 = conf1_flat > conf1_thresh
        mask2 = conf2_flat > conf2_thresh

        pts1_filtered = pts1_flat[mask1]
        pts2_filtered = pts2_flat[mask2]
        col1_filtered = col1_flat[mask1]
        col2_filtered = col2_flat[mask2]
        conf1_filtered = conf1_flat[mask1]
        conf2_filtered = conf2_flat[mask2]

        # Find correspondences
        tree = cKDTree(pts2_filtered)
        distances, indices = tree.query(pts1_filtered, k=1)

        # Compute scene scale for normalization
        scene_scale = np.percentile(distances, 90)

        # Change scores
        change_scores = distances / (scene_scale + 1e-6)

        # Classify changes
        static_mask = change_scores < change_threshold
        changed_mask = change_scores >= change_threshold

        return {
            'aligned_points_t1': pts1_filtered,
            'points_t2': pts2_filtered,
            'colors_t1': col1_filtered,
            'colors_t2': col2_filtered,
            'conf_t1': conf1_filtered,
            'conf_t2': conf2_filtered,
            'change_scores': change_scores,
            'static_mask': static_mask,
            'changed_mask': changed_mask,
            'correspondence_indices': indices,
            'correspondence_distances': distances,
            'scene_scale': scene_scale,
            'num_static': int(np.sum(static_mask)),
            'num_changed': int(np.sum(changed_mask)),
            'change_ratio': float(np.mean(changed_mask))
        }


def run_coarse_to_fine_registration(
    model,
    images_t1: List[str],
    images_t2: List[str],
    predictions_t1_dense: Dict,
    predictions_t2_dense: Dict,
    num_keyframes: int = 5,
    device: str = 'cuda'
) -> Dict:
    """
    Main function to run the two-stage registration pipeline.

    Args:
        model: VGGT model
        images_t1: List of T1 image paths
        images_t2: List of T2 image paths
        predictions_t1_dense: Dense predictions for T1 (full images)
        predictions_t2_dense: Dense predictions for T2 (full images)
        num_keyframes: Number of keyframes for coarse alignment
        device: Device to run on

    Returns:
        Dictionary with alignment results
    """
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    start_time = time.time()
    registrator = CoarseToFineRegistration(num_keyframes=num_keyframes)

    # =========== Stage 1: Coarse Alignment via Joint Reconstruction ===========
    print("=" * 60)
    print("Stage 1: Coarse Alignment via Joint Keyframe Reconstruction")
    print("=" * 60)

    # Select keyframes
    kf_indices_t1 = select_keyframes(images_t1, num_keyframes)
    kf_indices_t2 = select_keyframes(images_t2, num_keyframes)

    keyframe_paths_t1 = [images_t1[i] for i in kf_indices_t1]
    keyframe_paths_t2 = [images_t2[i] for i in kf_indices_t2]

    print(f"T1 keyframes: {kf_indices_t1}")
    print(f"T2 keyframes: {kf_indices_t2}")

    # Joint reconstruction
    all_keyframe_paths = keyframe_paths_t1 + keyframe_paths_t2
    print(f"Running joint reconstruction on {len(all_keyframe_paths)} keyframes...")

    images_joint = load_and_preprocess_images(all_keyframe_paths).to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions_joint = model(images_joint)

    # Process predictions
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions_joint['pose_enc'], images_joint.shape[-2:]
    )
    predictions_joint['extrinsic'] = extrinsic
    predictions_joint['intrinsic'] = intrinsic

    # Convert to numpy
    for key in predictions_joint.keys():
        if isinstance(predictions_joint[key], torch.Tensor):
            predictions_joint[key] = predictions_joint[key].cpu().numpy().squeeze(0)

    # Align intrinsics and compute world points
    intrinsic_ref = predictions_joint['intrinsic'][0]
    intrinsic_aligned, depth_aligned = align_intrinsics_and_depth(
        predictions_joint['intrinsic'],
        intrinsic_ref,
        predictions_joint['depth'][..., 0]
    )

    world_points_joint = unproject_depth_map_to_point_map(
        depth_aligned[..., None],
        predictions_joint['extrinsic'],
        intrinsic_aligned
    )
    predictions_joint['world_points_from_depth'] = world_points_joint

    # Get coarse alignment
    n_t1_kf = len(kf_indices_t1)
    n_t2_kf = len(kf_indices_t2)

    coarse_s, coarse_R, coarse_t = registrator.compute_coarse_alignment(
        predictions_joint, n_t1_kf, n_t2_kf
    )

    print(f"[Coarse] Initial scale: {coarse_s:.4f}")

    # =========== Compute scale from joint reconstruction ===========
    # The joint reconstruction provides relative poses between T1 and T2 frames
    # We can extract scale from comparing extrinsics

    # Get camera positions
    extrinsics_joint = predictions_joint['extrinsic']
    cam_positions = []
    for ext in extrinsics_joint:
        R_cam = ext[:3, :3]
        t_cam = ext[:3, 3]
        cam_pos = -R_cam.T @ t_cam
        cam_positions.append(cam_pos)
    cam_positions = np.array(cam_positions)

    # T1 camera positions and T2 camera positions
    cam_t1 = cam_positions[:n_t1_kf]
    cam_t2 = cam_positions[n_t1_kf:]

    # Use camera trajectory to estimate scale
    if len(cam_t1) > 1 and len(cam_t2) > 1:
        traj_scale_t1 = np.mean([np.linalg.norm(cam_t1[i+1] - cam_t1[i])
                                  for i in range(len(cam_t1)-1)])
        traj_scale_t2 = np.mean([np.linalg.norm(cam_t2[i+1] - cam_t2[i])
                                  for i in range(len(cam_t2)-1)])
        estimated_scale = traj_scale_t2 / (traj_scale_t1 + 1e-8)
        print(f"[Coarse] Trajectory-based scale estimate: {estimated_scale:.4f}")

    # =========== Stage 2: Fine Alignment using Static Regions ===========
    print("\n" + "=" * 60)
    print("Stage 2: Fine Alignment using Static Region Detection")
    print("=" * 60)

    # Get dense data
    points1_dense = predictions_t1_dense['world_points_from_depth']
    points2_dense = predictions_t2_dense['world_points_from_depth']
    conf1_dense = predictions_t1_dense['depth_conf']
    conf2_dense = predictions_t2_dense['depth_conf']
    depth1_dense = predictions_t1_dense['depth']
    depth2_dense = predictions_t2_dense['depth']

    # Handle depth shape
    if depth1_dense.ndim == 4:
        depth1_dense = depth1_dense[..., 0]
    if depth2_dense.ndim == 4:
        depth2_dense = depth2_dense[..., 0]

    # Compute initial alignment using joint reconstruction transform
    # For now, use the joint reconstruction to compute T1->T2 transform

    # Sample points from joint reconstruction for T1 and T2
    pts_t1_joint = world_points_joint[:n_t1_kf].reshape(-1, 3)
    pts_t2_joint = world_points_joint[n_t1_kf:].reshape(-1, 3)
    conf_t1_joint = predictions_joint['depth_conf'][:n_t1_kf].flatten()
    conf_t2_joint = predictions_joint['depth_conf'][n_t1_kf:].flatten()

    # High confidence sampling
    mask1 = conf_t1_joint > np.percentile(conf_t1_joint, 70)
    mask2 = conf_t2_joint > np.percentile(conf_t2_joint, 70)

    pts_t1_high = pts_t1_joint[mask1]
    pts_t2_high = pts_t2_joint[mask2]

    # Find correspondences in joint reconstruction
    if len(pts_t1_high) > 100 and len(pts_t2_high) > 100:
        # Subsample
        idx1 = np.random.choice(len(pts_t1_high), min(2000, len(pts_t1_high)), replace=False)
        idx2 = np.random.choice(len(pts_t2_high), min(2000, len(pts_t2_high)), replace=False)

        pts_t1_sub = pts_t1_high[idx1]
        pts_t2_sub = pts_t2_high[idx2]

        # Use joint reconstruction centroids for initial alignment
        centroid_t1 = np.mean(pts_t1_sub, axis=0)
        centroid_t2 = np.mean(pts_t2_sub, axis=0)

        # Build tree for matching
        tree = cKDTree(pts_t2_sub)
        distances, indices = tree.query(pts_t1_sub, k=1)

        # Use good correspondences (small distance)
        good_mask = distances < np.percentile(distances, 50)
        if np.sum(good_mask) > 50:
            src_pts = pts_t1_sub[good_mask]
            tgt_pts = pts_t2_sub[indices[good_mask]]

            # Compute initial Sim(3) from joint reconstruction correspondences
            coarse_s, coarse_R, coarse_t = umeyama_sim3(src_pts, tgt_pts, with_scale=True)
            print(f"[Coarse] Sim(3) from joint correspondences: scale={coarse_s:.4f}")

    # Refine alignment using dense reconstructions
    fine_s, fine_R, fine_t, refine_info = registrator.refine_alignment(
        points1_dense, conf1_dense, depth1_dense,
        points2_dense, conf2_dense, depth2_dense,
        coarse_s, coarse_R, coarse_t
    )

    print(f"\n[Fine] Final scale: {fine_s:.4f}")
    print(f"[Fine] Final error: {refine_info['final_error']:.6f}")

    # =========== Change Detection ===========
    print("\n" + "=" * 60)
    print("Change Detection")
    print("=" * 60)

    # Get colors
    images1 = predictions_t1_dense.get('images', np.ones_like(points1_dense))
    images2 = predictions_t2_dense.get('images', np.ones_like(points2_dense))

    if images1.ndim == 4 and images1.shape[1] == 3:
        colors1 = np.transpose(images1, (0, 2, 3, 1))
    else:
        colors1 = images1
    if images2.ndim == 4 and images2.shape[1] == 3:
        colors2 = np.transpose(images2, (0, 2, 3, 1))
    else:
        colors2 = images2

    colors1 = (colors1 * 255).astype(np.uint8)
    colors2 = (colors2 * 255).astype(np.uint8)

    change_results = registrator.detect_changes(
        points1_dense, conf1_dense, colors1,
        points2_dense, conf2_dense, colors2,
        fine_s, fine_R, fine_t
    )

    print(f"Static points: {change_results['num_static']}")
    print(f"Changed points: {change_results['num_changed']}")
    print(f"Change ratio: {change_results['change_ratio']*100:.1f}%")

    elapsed_time = time.time() - start_time

    # =========== Compile Results ===========
    return {
        'method': 'VGGT-CD (Coarse-to-Fine)',
        # Coarse stage results
        'coarse_scale': coarse_s,
        'coarse_rotation': coarse_R,
        'coarse_translation': coarse_t,
        # Fine stage results
        'scale': fine_s,
        'rotation': fine_R,
        'translation': fine_t,
        'refine_info': refine_info,
        # Change detection
        'change_results': change_results,
        # Full point clouds
        'source_points': points1_dense.reshape(-1, 3),
        'target_points': points2_dense.reshape(-1, 3),
        'aligned_points': apply_sim3(points1_dense, fine_s, fine_R, fine_t).reshape(-1, 3),
        'source_colors': colors1.reshape(-1, 3),
        'target_colors': colors2.reshape(-1, 3),
        'source_conf': conf1_dense.flatten(),
        'target_conf': conf2_dense.flatten(),
        # Timing
        'elapsed_time': elapsed_time
    }
