import csv
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "experiments" / "results" / "summary.csv"
SUMMARY_CHART = ROOT / "docs" / "assets" / "metrics-overview.png"


def load_summary_rows():
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def summary_table():
    rows = load_summary_rows()
    headers = ["scene", "N_frames", "ATE_Joint", "ATE_rmse", "RTE_mean", "RRE_mean", "Time(s)", "Mem(MB)"]
    table = []
    for row in rows:
        table.append([row.get(h, "") for h in headers])
    return headers, table


def scene_report(scene):
    rows = load_summary_rows()
    if not rows:
        return "No bundled summary.csv was found."
    row = next((item for item in rows if item["scene"] == scene), rows[0])
    return (
        f"### {row['scene']}\n"
        f"- Frames: **{row['N_frames']}**\n"
        f"- ATE mean: **{float(row['ATE_Joint']):.4f}**\n"
        f"- ATE RMSE: **{float(row['ATE_rmse']):.4f}**\n"
        f"- RTE mean: **{float(row['RTE_mean']):.4f}**\n"
        f"- RRE mean: **{float(row['RRE_mean']):.4f}**\n"
        f"- Runtime: **{float(row['Time(s)']):.2f}s**\n"
        f"- GPU memory: **{float(row['Mem(MB)']):.2f} MB**"
    )


def run_uploaded_zip(data_zip, scene, seq_t1, seq_t2, num_keyframes, conf_threshold):
    model_path = Path(os.environ.get("VGGT_MODEL_PATH", ROOT / "model.pt")).expanduser()
    if not model_path.exists():
        return (
            "Model checkpoint was not found. Set VGGT_MODEL_PATH on the server or place model.pt in the app root.",
            None,
        )
    if data_zip is None:
        return "Upload a ZIP file containing the DATA_ROOT scene layout first.", None

    timeout = int(os.environ.get("VGGT_DEMO_TIMEOUT", "1800"))
    with tempfile.TemporaryDirectory(prefix="vggt_cd_demo_") as tmp:
        tmp_dir = Path(tmp)
        data_root = tmp_dir / "data"
        output_root = tmp_dir / "outputs"
        data_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(data_root)

        cmd = [
            sys.executable,
            str(ROOT / "run_inference.py"),
            "--data_root",
            str(data_root),
            "--output_root",
            str(output_root),
            "--model_path",
            str(model_path),
            "--num_keyframes",
            str(int(num_keyframes)),
            "--conf_threshold",
            str(int(conf_threshold)),
        ]
        if scene:
            cmd.extend(["--scene", scene.strip()])
        if seq_t1:
            cmd.extend(["--seq_t1", seq_t1.strip()])
        if seq_t2:
            cmd.extend(["--seq_t2", seq_t2.strip()])

        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        log = (proc.stdout + "\n" + proc.stderr).strip()
        if proc.returncode != 0:
            return "Inference failed:\n\n```text\n" + log[-6000:] + "\n```", None

        eval_cmd = [
            sys.executable,
            str(ROOT / "experiments" / "evaluation" / "eval_poses_1.py"),
            "--eval_root",
            str(output_root),
        ]
        eval_proc = subprocess.run(
            eval_cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        eval_log = (eval_proc.stdout + "\n" + eval_proc.stderr).strip()
        summary_path = output_root / "summary.csv"
        if summary_path.exists():
            exported = ROOT / "demo_summary.csv"
            shutil.copy2(summary_path, exported)
            return "Inference completed.\n\n```text\n" + eval_log[-6000:] + "\n```", str(exported)
        return "Inference completed, but no summary.csv was generated.\n\n```text\n" + eval_log[-6000:] + "\n```", None


rows = load_summary_rows()
scene_names = [row["scene"] for row in rows] or ["no-summary"]
headers, table = summary_table()

with gr.Blocks(title="VGGT-CD Demo") as demo:
    gr.Markdown(
        "# VGGT-CD\n"
        "Explore bundled benchmark results immediately, or run the full pipeline on a hosted GPU Space "
        "after configuring `VGGT_MODEL_PATH`."
    )

    with gr.Tab("Sample Results"):
        scene = gr.Dropdown(scene_names, value=scene_names[0], label="Scene")
        report = gr.Markdown(scene_report(scene_names[0]))
        chart = gr.Image(value=str(SUMMARY_CHART), label="Summary Chart", type="filepath")
        gr.Dataframe(value=table, headers=headers, label="Bundled summary.csv", interactive=False)
        scene.change(scene_report, inputs=scene, outputs=report)

    with gr.Tab("Run on Your ZIP"):
        gr.Markdown(
            "Upload a ZIP that contains the expected DATA_ROOT layout. This tab needs a server with GPU "
            "and a configured VGGT checkpoint."
        )
        data_zip = gr.File(label="DATA_ROOT ZIP", file_types=[".zip"], type="filepath")
        with gr.Row():
            run_scene = gr.Textbox(label="Scene", placeholder="breville")
            seq_t1 = gr.Textbox(label="T1 sequence", placeholder="IMG_9184")
            seq_t2 = gr.Textbox(label="T2 sequence", placeholder="IMG_9185")
        with gr.Row():
            num_keyframes = gr.Slider(2, 12, value=5, step=1, label="Number of keyframes")
            conf_threshold = gr.Slider(0, 100, value=50, step=1, label="Confidence percentile")
        run_btn = gr.Button("Run VGGT-CD", variant="primary")
        run_log = gr.Markdown()
        run_file = gr.File(label="summary.csv")
        run_btn.click(
            run_uploaded_zip,
            inputs=[data_zip, run_scene, seq_t1, seq_t2, num_keyframes, conf_threshold],
            outputs=[run_log, run_file],
        )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
    )
