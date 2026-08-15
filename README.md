# VGGT-CD: Training-Free Robust Registration for 3D Change Detection

<p align="center">
  <strong><a href="https://arxiv.org/abs/2605.16859">Paper</a></strong> |
  <strong><a href="https://github.com/WZ-CS/VGGT-CD">Code</a></strong>
</p>

Official implementation of **VGGT-CD**, a training-free coarse-to-fine pipeline for 3D change detection from unposed multi-view images.

**Abstract:** Independent reconstructions of two time epochs suffer from scale ambiguity, the registration-change paradox, and edge-flying noise. VGGT-CD uses sparse joint inference to establish a shared metric space and an initial Sim(3) prior, then refines dense reconstructions through reliability-guided purification and closed-form translation alignment. On the 11-scene World Across Time benchmark, it reduces absolute trajectory error by 44% outdoors and 59% indoors while completing registration more than 6x faster than prior baselines.

<p align="center">
  <img src="assets/vggt-cd-overview.png" alt="VGGT-CD overview" width="100%">
</p>

<p align="center"><em>Figure 1. VGGT-CD aligns independent bi-temporal reconstructions and isolates genuine 3D changes, resolving scale ambiguity, dynamic outliers, and edge noise.</em></p>

## Method

VGGT-CD addresses the coordinate and scale inconsistency between independently reconstructed T1/T2 image sequences.

1. VGGT reconstructs dense geometry, depth, confidence, and camera poses for T1 and T2.
2. A sparse joint-inference stage estimates a cross-time Sim(3) transform in a unified metric space.
3. High-confidence dense points are filtered to remove dynamic changes and edge noise.
4. Static correspondences are refined with voxel hashing and one-shot SVD.
5. The final alignment is saved as `fine_sim3.npz`.

## Repository Structure

```text
VGGT-CD/
|- run_inference.py               # Main VGGT-CD pipeline
|- run_inference.sh               # Shell wrapper for inference
|- coarse_to_fine_registration.py # Core coarse-to-fine registration method
|- IMPROVEMENTS.md                # Method details
|- app.py                         # Gradio demo entry
|- vggt/                          # VGGT model code
|- experiments/
|  |- baselines/                 # Baseline methods
|  |- evaluation/                # Evaluation scripts
|  `- results/summary.csv        # Experiment summary
|- docs/                          # GitHub Pages project site
`- requirements.txt
```

## Installation

```bash
git clone https://github.com/WZ-CS/VGGT-CD.git
cd VGGT-CD

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the VGGT checkpoint path:

```bash
export VGGT_MODEL_PATH=/path/to/model.pt
```

## Data Format

```text
DATA_ROOT/
`- scene_name/
    |- images/
    |  |- SEQ_T1/
    |  `- SEQ_T2/
    `- sparse/0/
        |- cameras.bin
        |- images.bin
        `- points3D.bin
```

## Inference

```bash
python run_inference.py \
  --data_root /path/to/WAT \
  --scene breville \
  --seq_t1 IMG_9184 \
  --seq_t2 IMG_9185 \
  --output_root ./eval_results \
  --model_path "${VGGT_MODEL_PATH}" \
  --num_keyframes 5 \
  --conf_threshold 50
```

Shell wrapper:

```bash
DATA_ROOT=/path/to/WAT \
SCENE=breville \
SEQ_T1=IMG_9184 \
SEQ_T2=IMG_9185 \
VGGT_MODEL_PATH=/path/to/model.pt \
./run_inference.sh
```

## Evaluation

```bash
python experiments/evaluation/eval_poses_1.py --eval_root ./eval_results
```

Scene outputs:

```text
eval_results/scene_name/
|- pred_poses_t1.npy
|- pred_poses_t2.npy
|- gt_poses_t1.npy
|- gt_poses_t2.npy
|- image_names_t1.txt
|- image_names_t2.txt
|- coarse_sim3.npz
|- fine_sim3.npz
`- timing.json
```

## Results

| Scenes | ATE mean | ATE RMSE | RTE mean | RRE mean | Time/scene | GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.4316 | 0.4631 | 0.1933 | 0.3598 | 33.14 s | 18167.52 MB |

## Baselines

| Script | Method |
| --- | --- |
| `experiments/baselines/run_baseline_vggt_only.py` | Independent VGGT T1/T2 inference |
| `experiments/baselines/baseline_icp.py` | VGGT point clouds with ICP |
| `experiments/baselines/run_baseline_fgr.py` | VGGT point clouds with Fast Global Registration |
| `experiments/baselines/run_baseline_global_scale_icp.py` | Global scale alignment with ICP |
| `experiments/baselines/run_baseline_sim3_icp.py` | Sim(3) alignment with ICP |
| `experiments/baselines/run_baseline_geotransformer.py` | GeoTransformer baseline |

## Demo

```bash
GRADIO_SERVER_PORT=7860 python app.py
```
