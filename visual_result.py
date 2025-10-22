import numpy as np
from PIL import Image
import os
from pathlib import Path


def visualize_batch_labels(
        pred_label_dir: str,  # 已推理出的标签图文件夹（如inference_results）
        save_visual_dir: str,  # 彩色可视化结果保存文件夹（自动创建）
        color_map: dict = None  # 自定义颜色映射
):
    """
    批量处理所有推理标签图，生成彩色可视化结果
    """
    # 1. 默认颜色映射
    if color_map is None:
        color_map = {
            0: (255, 255, 255),  # 背景 → 黑色
            1: (255, 255, 255),  # dc0 → 红色
            2: (0, 255, 255)  # dc1 → 蓝色
        }

    # 2. 确保保存目录存在
    os.makedirs(save_visual_dir, exist_ok=True)

    # 3. 遍历标签图文件夹（仅处理PNG文件，跳过其他格式）
    pred_label_paths = list(Path(pred_label_dir).glob("*.png"))
    if len(pred_label_paths) == 0:
        print(f"警告：在 {pred_label_dir} 中未找到PNG标签图，请检查路径！")
        return

    # 4. 批量处理每张标签图
    for idx, pred_path in enumerate(pred_label_paths, 1):
        # 获取标签图文件名（如test_case001.png → 文件名：test_case001）
        case_name = pred_path.stem

        # 读取标签图
        pred_label = Image.open(pred_path)
        pred_label_np = np.array(pred_label, dtype=np.uint16)

        # 检查非法标签值
        unique_labels = np.unique(pred_label_np)
        for label in unique_labels:
            if label not in color_map.keys():
                print(f"跳过 {case_name}：存在未定义标签值 {label}，请更新color_map")
                continue

        # 标签值→彩色图
        pred_visual = np.zeros((pred_label_np.shape[0], pred_label_np.shape[1], 3), dtype=np.uint8)
        for label, color in color_map.items():
            pred_visual[pred_label_np == label] = color

        # 保存彩色图（文件名：原文件名_color.png）
        save_path = Path(save_visual_dir) / f"{case_name}_color.png"
        Image.fromarray(pred_visual).save(save_path, format='PNG')

        # 打印进度
        print(f"已处理 {idx}/{len(pred_label_paths)}：{case_name} → 保存至 {save_path}")

    print(f"\n批量可视化完成！共处理 {len(pred_label_paths)} 张标签图，结果目录：{save_visual_dir}")


# ------------------- 请替换为你的实际路径 -------------------
if __name__ == "__main__":
    # 已推理出的标签图文件夹（必填！如nnUNetv2_predict的-o参数路径）
    PRED_LABEL_DIR = r"F:\CJY\deep-learning\pytorch-CycleGAN-and-pix2pi\imgDataset\test\inference_results"
    # 彩色可视化结果保存文件夹（自定义，自动创建）
    SAVE_VISUAL_DIR = r"F:\CJY\deep-learning\pytorch-CycleGAN-and-pix2pi\imgDataset\test\visualization_batch2"

    # 运行批量可视化（如需自定义颜色，可传入color_map，示例：
    custom_color = {0:(255,255,255), 1:(255,0,0), 2:(0,255,0)}  # 背景白色、dc1绿色
    visualize_batch_labels(PRED_LABEL_DIR, SAVE_VISUAL_DIR, custom_color)
    # ）
    #visualize_batch_labels(PRED_LABEL_DIR, SAVE_VISUAL_DIR)