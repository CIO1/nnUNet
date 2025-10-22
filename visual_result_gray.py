import numpy as np
from PIL import Image
import os
from pathlib import Path


def visualize_batch_labels_gray(
        pred_label_dir: str,  # 已推理出的标签图文件夹（如inference_results）
        save_visual_dir: str,  # 灰度可视化结果保存文件夹（自动创建）
        gray_map: dict = None  # 自定义灰度映射（键为标签，值为0-255的灰度值）
):
    """
    批量处理所有推理标签图，生成灰度可视化结果
    """
    # 1. 默认灰度映射（0-255之间的灰度值，数值越大越亮）
    if gray_map is None:
        gray_map = {
            0: 255,   # 背景 → 白色（最亮）
            1: 0,   # dc0 → 中灰色
            2: 128     # dc1 → 深灰色（较暗）
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
        # 获取标签图文件名
        case_name = pred_path.stem

        # 读取标签图
        pred_label = Image.open(pred_path)
        pred_label_np = np.array(pred_label, dtype=np.uint16)

        # 检查非法标签值
        unique_labels = np.unique(pred_label_np)
        invalid_labels = [label for label in unique_labels if label not in gray_map.keys()]
        if invalid_labels:
            print(f"跳过 {case_name}：存在未定义标签值 {invalid_labels}，请更新gray_map")
            continue

        # 标签值→灰度图（单通道）
        pred_visual = np.zeros(pred_label_np.shape, dtype=np.uint8)  # 单通道数组
        for label, gray_value in gray_map.items():
            pred_visual[pred_label_np == label] = gray_value

        # 保存灰度图（文件名：原文件名_gray.png）
        save_path = Path(save_visual_dir) / f"{case_name[:-3]}{case_name[-3:]}_mask.png"
        Image.fromarray(pred_visual).save(save_path, format='PNG')

        # 打印进度
        print(f"已处理 {idx}/{len(pred_label_paths)}：{case_name} → 保存至 {save_path}")

    print(f"\n批量灰度可视化完成！共处理 {len(pred_label_paths)} 张标签图，结果目录：{save_visual_dir}")


# ------------------- 请替换为你的实际路径 -------------------
if __name__ == "__main__":
    # 已推理出的标签图文件夹（必填）
    PRED_LABEL_DIR = r"F:\ImageDataSet\mask"
    # 灰度可视化结果保存文件夹（自动创建）
    SAVE_VISUAL_DIR = r"F:\ImageDataSet\mask_visual"

    # 运行批量可视化（如需自定义灰度，可传入gray_map，示例：
    # custom_gray = {0:255, 1:100, 2:50}  # 背景白色、dc0为100、dc1为50
    # visualize_batch_labels_gray(PRED_LABEL_DIR, SAVE_VISUAL_DIR, custom_gray)
    # ）
    visualize_batch_labels_gray(PRED_LABEL_DIR, SAVE_VISUAL_DIR)