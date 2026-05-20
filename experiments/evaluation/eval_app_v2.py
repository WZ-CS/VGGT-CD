import os
import open3d as o3d
import numpy as np
import gradio as gr
import trimesh
import matplotlib.pyplot as plt
import tempfile

def export_ply_with_rgba(points, colors, filename):
    """
    将点云及 RGBA 颜色导出为 PLY 文件
    :param points: (N, 3) numpy array
    :param colors: (N, 4) numpy array, uint8 (包含 Alpha 透明通道)
    """
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    filepath = os.path.join(tempfile.gettempdir(), filename)
    cloud.export(file_obj=filepath)
    return filepath

def evaluate_and_visualize(file_a, file_b, inlier_threshold, max_heatmap_dist, subsample_ratio):
    """
    计算配准指标并生成可视化结果
    :param file_a: Target 点云 (基准)
    :param file_b: Aligned Source 点云 (待评估)
    """
    if not file_a or not file_b:
        return "请上传两个点云文件。", None, None

    try:
        # 1. 加载点云
        # 兼容 Gradio 返回的文件对象或路径
        path_a = file_a.name if hasattr(file_a, 'name') else str(file_a)
        path_b = file_b.name if hasattr(file_b, 'name') else str(file_b)

        pcd_a = o3d.io.read_point_cloud(path_a)
        pcd_b = o3d.io.read_point_cloud(path_b)

        if len(pcd_a.points) == 0 or len(pcd_b.points) == 0:
            return "错误：未能成功读取点云数据，请检查格式 (支持 .ply, .pcd, .obj)。", None, None

        # 如果点数过多，为了可视化不卡顿进行随机降采样
        if subsample_ratio < 1.0:
            pcd_a = pcd_a.random_down_sample(subsample_ratio)
            pcd_b = pcd_b.random_down_sample(subsample_ratio)

        # 2. 计算点到点距离 (B 到 A 的最近邻距离)
        distances = pcd_b.compute_point_cloud_distance(pcd_a)
        distances = np.asarray(distances)

        # 3. 计算量化指标
        inlier_mask = distances < inlier_threshold
        fitness = np.sum(inlier_mask) / len(distances)
        
        if np.sum(inlier_mask) > 0:
            inlier_rmse = np.sqrt(np.mean(distances[inlier_mask] ** 2))
        else:
            inlier_rmse = float('inf')
            
        mean_dist = np.mean(distances)

        # 构建报告
        report = f"""
        ### 📊 量化评估报告 (Quantitative Metrics)
        
        * **Fitness (内点率 / 重合度)**: `{fitness * 100:.2f}%` 
            *(在阈值 {inlier_threshold} 内匹配成功的点占比。越高越好。)*
        * **Inlier RMSE (内点均方根误差)**: `{inlier_rmse:.6f}`
            *(成功匹配的点之间的平均误差。越低越好。)*
        * **Mean Distance (全局平均距离)**: `{mean_dist:.6f}`
        
        > **提示**: 点云 B 共 `{len(distances)}` 个点，点云 A 共 `{len(pcd_a.points)}` 个点。
        """

        # ==========================================
        # 4. 生成误差热力图 (Error Heatmap) -> 导出为 PLY
        # ==========================================
        norm_distances = np.clip(distances / max_heatmap_dist, 0, 1)
        cmap = plt.get_cmap("jet")
        # cmap 默认返回 (N, 4) 的 RGBA 浮点数组 [0~1]，我们直接转换为 0~255 的 uint8
        heatmap_colors = (cmap(norm_distances) * 255).astype(np.uint8)
        heatmap_points = np.asarray(pcd_b.points)
        
        heatmap_ply = export_ply_with_rgba(heatmap_points, heatmap_colors, "heatmap.ply")

        # ==========================================
        # 5. 生成叠加对比图 (Overlay) -> 导出为 PLY
        # ==========================================
        # 处理点云 A：设置为半透明红色
        points_a = np.asarray(pcd_a.points)
        colors_a = np.zeros((len(points_a), 4), dtype=np.uint8)
        colors_a[:, 0] = 255  # 红色通道打满
        colors_a[:, 3] = 100  # Alpha 通道设为 100 (0为全透明，255为不透明，可根据需要微调)

        # 处理点云 B：保持原色或设为不透明蓝色
        points_b = np.asarray(pcd_b.points)
        if pcd_b.has_colors():
            colors_b_rgb = (np.asarray(pcd_b.colors) * 255).astype(np.uint8)
        else:
            colors_b_rgb = np.tile([25, 127, 204], (len(points_b), 1)).astype(np.uint8) # 默认蓝色
            
        # 给点云 B 补充不透明的 Alpha 通道
        alpha_b = np.full((len(points_b), 1), 255, dtype=np.uint8)
        colors_b = np.concatenate([colors_b_rgb, alpha_b], axis=1)

        # 合并顶点和颜色
        overlay_points = np.vstack((points_a, points_b))
        overlay_colors = np.vstack((colors_a, colors_b))
        
        overlay_ply = export_ply_with_rgba(overlay_points, overlay_colors, "overlay.ply")

        return report, heatmap_ply, overlay_ply

    except Exception as e:
        import traceback
        error_msg = f"发生错误: {str(e)}\n{traceback.format_exc()}"
        return error_msg, None, None


# ==========================================
# 构建 Gradio 界面
# ==========================================
with gr.Blocks(title="Point Cloud Registration Evaluator") as demo:
    gr.Markdown("""
    # 🎯 点云配准质量评估工具 (Registration Evaluator)
    上传基准点云 (Target) 和已配准的点云 (Aligned Source)，一键生成量化指标与误差热力图。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_a = gr.File(label="上传基准点云 A (Target)", file_types=[".ply", ".pcd", ".obj"])
            file_b = gr.File(label="上传配准后的点云 B (Aligned Source)", file_types=[".ply", ".pcd", ".obj"])
            
            gr.Markdown("### ⚙️ 评估参数设置")
            inlier_thresh = gr.Slider(minimum=0.01, maximum=2.0, value=0.15, step=0.01, 
                                      label="Inlier Threshold (Fitness & RMSE 的距离阈值)", 
                                      info="判断两个点是否重合的最大距离。")
            heatmap_max = gr.Slider(minimum=0.01, maximum=5.0, value=0.5, step=0.01, 
                                    label="Heatmap Max Distance (热力图红色的最大距离)", 
                                    info="距离大于该值的点将被渲染为深红色。")
            subsample = gr.Slider(minimum=0.01, maximum=1.0, value=1.0, step=0.01,
                                  label="可视化降采样率", info="如果点云过大导致浏览器卡顿，请调小此值。")
            
            btn = gr.Button("开始评估 (Evaluate)", variant="primary")

        with gr.Column(scale=2):
            report_out = gr.Markdown("📊 等待上传与计算...")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔥 误差热力图 (Error Heatmap)\n*(蓝=误差小，红=误差大)*")
            heatmap_out = gr.Model3D(label="Heatmap (PLY)", height=500)
        with gr.Column():
            gr.Markdown("### 🔗 结构叠加图 (Overlay)\n*(半透明红色=点云 A, 原色/蓝色=点云 B)*")
            overlay_out = gr.Model3D(label="Overlay (PLY)", height=500)

    btn.click(
        fn=evaluate_and_visualize,
        inputs=[file_a, file_b, inlier_thresh, heatmap_max, subsample],
        outputs=[report_out, heatmap_out, overlay_out]
    )

demo.launch(
    server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7870")),
    share=os.environ.get("GRADIO_SHARE", "0") == "1",
)
