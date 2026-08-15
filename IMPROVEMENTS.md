# VGGT-CD Improvements

This file separates the main method contribution from the experiment scripts.

## Core Idea

VGGT independently reconstructs image sequences, but T1 and T2 predictions can live in different coordinate systems and scales. VGGT-CD adds a cross-time registration stage so the two reconstructions can be compared in one coordinate frame.

## Coarse Stage

`coarse_to_fine_registration.py` selects keyframes from T1 and T2, reconstructs them jointly, and estimates an initial Sim(3) transform:

```text
T1 points -- Sim(3): scale, rotation, translation --> T2 frame
```

This stage gives a robust global alignment before dense refinement.

## Fine Stage

The fine stage filters high-confidence dense points, applies the coarse Sim(3), finds likely static correspondences with voxel hashing, and solves a one-shot SVD/Umeyama refinement.

The output is saved as:

```text
fine_sim3.npz
├── s
├── R
└── t
```

## Main Files

- `run_inference.py`: end-to-end VGGT-CD inference.
- `coarse_to_fine_registration.py`: Sim(3), keyframe selection, coarse alignment, and fine registration utilities.
- `experiments/`: baselines, ablations, evaluation scripts, and result summaries.
