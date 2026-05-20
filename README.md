# VGGT-CD

[Project Page](https://YOUR_GITHUB_USERNAME.github.io/VGGT-CD/) | [Online Demo](https://huggingface.co/spaces/YOUR_HF_USERNAME/VGGT-CD)

VGGT-CD is a coarse-to-fine bi-temporal point cloud registration and change detection pipeline built on VGGT. It reconstructs two image sequences from different times, estimates a cross-time Sim(3) alignment, refines static correspondences, and exports pose/registration metrics.

## What Is In This Release

This folder is the clean GitHub-ready version. Large local files and paper drafts are intentionally excluded.

- Core method and improvements are kept at the repository root.
- Experiment, baseline, evaluation, and result files are grouped under `experiments/`.
- Website files are under `docs/` for GitHub Pages.
- `app.py` is prepared for a Hugging Face Spaces demo link.
- `model.pt`, datasets, generated outputs, and the ECCV PDF are not included.

## Repository Layout

```text
VGGT-CD/
├── app.py                         # Hugging Face Spaces / Gradio demo entry
├── IMPROVEMENTS.md                # Method contribution separated from experiments
├── run_inference.py               # Main VGGT-CD pipeline
├── run_inference.sh               # Configurable shell wrapper
├── coarse_to_fine_registration.py # Main coarse-to-fine registration improvement
├── vggt/                          # Local VGGT model code
├── experiments/
│   ├── baselines/                 # VGGT-only, ICP, FGR, GeoTransformer baselines
│   ├── evaluation/                # Evaluation and timing scripts
│   └── results/summary.csv        # Bundled experiment summary
├── docs/                          # Static GitHub Pages website
├── requirements.txt
└── .github/workflows/pages.yml
```

## Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/VGGT-CD.git
cd VGGT-CD

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the VGGT checkpoint at `./model.pt`, or set:

```bash
export VGGT_MODEL_PATH=/path/to/model.pt
```

The checkpoint is intentionally ignored by git because it is too large for a normal GitHub repository.

## Data Format

```text
DATA_ROOT/
└── scene_name/
    ├── images/
    │   ├── SEQ_T1/
    │   └── SEQ_T2/
    └── sparse/0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Run VGGT-CD

```bash
python run_inference.py \
  --data_root /path/to/WAT \
  --scene breville \
  --seq_t1 IMG_9184 \
  --seq_t2 IMG_9185 \
  --output_root ./eval_results \
  --model_path "$VGGT_MODEL_PATH" \
  --num_keyframes 5 \
  --conf_threshold 50
```

Or:

```bash
DATA_ROOT=/path/to/WAT \
SCENE=breville \
SEQ_T1=IMG_9184 \
SEQ_T2=IMG_9185 \
VGGT_MODEL_PATH=/path/to/model.pt \
./run_inference.sh
```

## Evaluate Results

```bash
python experiments/evaluation/eval_poses_1.py --eval_root ./eval_results
```

Each scene output contains predicted poses, ground-truth poses, `coarse_sim3.npz`, `fine_sim3.npz`, image names, and `timing.json`.

## Experiments

Baselines live in `experiments/baselines/`:

| Script | Method |
| --- | --- |
| `run_baseline_vggt_only.py` | Independent VGGT T1/T2 inference without registration |
| `baseline_icp.py` | VGGT point clouds plus ICP |
| `run_baseline_fgr.py` | VGGT point clouds plus Fast Global Registration |
| `run_baseline_global_scale_icp.py` | Global scale alignment plus ICP |
| `run_baseline_sim3_icp.py` | Sim(3) alignment plus ICP |
| `run_baseline_geotransformer.py` | External GeoTransformer baseline |

The bundled `experiments/results/summary.csv` contains 11 WAT scenes:

| Scenes | ATE mean | ATE RMSE | RTE mean | RRE mean | Time/scene | GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.4316 | 0.4631 | 0.1933 | 0.3598 | 33.14 s | 18167.52 MB |

## Website And Online Demo

There are two different web targets:

- `docs/`: static GitHub Pages project website. It can show the method, commands, and bundled results.
- `app.py`: Gradio app for Hugging Face Spaces or your own GPU server. This is the part that gives people a public link to try the demo.

GitHub Pages cannot run Python/GPU inference. To let anyone try the effect through a website link, deploy this repository or `app.py` to Hugging Face Spaces, choose a GPU runtime, upload or mount `model.pt`, and set `VGGT_MODEL_PATH`.

Run the demo locally:

```bash
GRADIO_SERVER_PORT=7860 python app.py
```

After deployment, replace the demo link at the top of this README and in `docs/index.html`.

## GitHub Pages

The workflow in `.github/workflows/pages.yml` publishes `docs/` automatically after pushing to `main`.

When creating the GitHub repository, do not add a README, `.gitignore`, or license through the GitHub UI if you plan to upload this prepared folder. This release already includes README and `.gitignore`.

## Notes Before Public Release

- Replace `YOUR_GITHUB_USERNAME` and `YOUR_HF_USERNAME`.
- Add a final license after confirming third-party VGGT licensing requirements.
- Keep `model.pt`, datasets, generated outputs, and paper PDFs outside the git history.
