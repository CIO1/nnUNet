import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2  # 用于多边形填充


def labelme_json_to_mask(
        input_json_dir: str,
        output_mask_dir: str,
        manual_image_size: tuple = None  # 手动指定图像尺寸：(width, height)，无则自动提取
) -> None:
    """
    将LabelMe风格JSON转为16位PNG Mask，标签映射规则：
    - 无标签区域（背景）→ 0
    - label="0" → 1
    - label="1" → 2
    - label="n" → n+1（以此类推）

    Args:
        input_json_dir: 输入JSON文件夹（存放casexxx_0000.json）
        output_mask_dir: 输出Mask文件夹（保存casexxx.png）
        manual_image_size: 手动指定图像尺寸（宽, 高），如(1024, 768)；None则自动提取
    """
    # 路径初始化
    input_dir = Path(input_json_dir)
    output_dir = Path(output_mask_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出文件夹：{output_dir}")

    # 遍历符合命名规则的JSON文件（casexxx_0000.json）
    json_files = list(input_dir.glob("case[0-9][0-9][0-9]_0000.json"))
    if not json_files:
        print("未找到符合规则的JSON文件（命名需为casexxx_0000.json）")
        return
    print(f"发现 {len(json_files)} 个JSON文件，开始处理...\n")

    # 逐个处理JSON
    for json_path in json_files:
        try:
            # 生成输出文件名（casexxx_0000.json → casexxx.png）
            json_filename = json_path.name
            case_id = json_filename.replace("_0000.json", "")
            output_mask_path = output_dir / f"{case_id}.tif"

            # 读取JSON数据
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            print(f"处理文件：{json_filename}")

            # 获取图像尺寸
            if manual_image_size:
                img_width, img_height = manual_image_size
                print(f"使用手动尺寸：宽={img_width}, 高={img_height}")
            else:
                try:
                    img_width = int(json_data["imageWidth"])
                    img_height = int(json_data["imageHeight"])
                    print(f"自动提取尺寸：宽={img_width}, 高={img_height}")
                except KeyError:
                    raise ValueError(
                        f"JSON缺少imageWidth/imageHeight，请手动指定manual_image_size=(宽, 高)"
                    )

            # 初始化Mask（背景为0，16位无符号整数）
            mask = np.zeros((img_height, img_width), dtype=np.uint16)

            # 处理每个多边形标注
            shapes = json_data.get("shapes", [])
            if not shapes:
                print(f"警告：{json_filename} 无标注区域，生成全背景Mask（全0）")
            else:
                for idx, shape in enumerate(shapes):
                    # 提取并转换原始标签（字符串→整数）
                    raw_label_str = shape["label"]
                    try:
                        raw_label = int(raw_label_str)
                    except ValueError:
                        raise ValueError(f"区域{idx + 1}的label={raw_label_str}不是有效整数，跳过")

                    # 计算映射后的标签（n→n+1）
                    mapped_label = raw_label + 1
                    if mapped_label > 65535:  # 16位无符号整数上限
                        raise ValueError(f"映射后标签{mapped_label}超过16位上限（65535）")

                    # 提取多边形坐标并转换格式
                    points = shape["points"]
                    if len(points) < 3:
                        print(f"区域{idx + 1}顶点不足3个（{len(points)}个），跳过")
                        continue
                    contour = np.array(points, dtype=np.int32).reshape(-1, 2)  # 适配OpenCV格式

                    # 填充多边形区域为映射后的标签
                    cv2.fillPoly(mask, [contour], color=mapped_label)
                    print(f"  - 区域{idx + 1}：原始label={raw_label} → 映射后={mapped_label}，顶点数={len(points)}")

            # 保存为16位PNG
            mask_image = Image.fromarray(mask, mode="I;16")  # 16位无符号整数模式
            mask_image.save(output_mask_path, format="TIFF", bits=16)
            print(f"成功保存：{output_mask_path.name}\n")

        except Exception as e:
            print(f"处理 {json_filename} 失败：{str(e)}\n")
            continue

    print(f"处理完成！输出路径：{output_dir}")

if __name__ == "__main__":
    # -------------------------- 请根据你的实际情况修改以下参数 --------------------------
    INPUT_JSON_DIR = r"F:\ImageDataSet\RawData\output_results_all_torch" # 输入JSON文件夹（存casexxx_0000.json）
    OUTPUT_MASK_DIR = r"F:\ImageDataSet\RawData\mask"  # 输出Mask文件夹（存casexxx.png）
    # 图像尺寸：优先从JSON自动提取；若JSON无imageWidth/imageHeight，需手动指定（例：(1024, 768)）
    MANUAL_IMAGE_SIZE = None  # 手动指定格式：(宽度, 高度)，如MANUAL_IMAGE_SIZE = (844, 459)
    # -----------------------------------------------------------------------------------

    # 调用函数执行转换
    labelme_json_to_mask(
        input_json_dir=INPUT_JSON_DIR,
        output_mask_dir=OUTPUT_MASK_DIR,
        manual_image_size=MANUAL_IMAGE_SIZE
    )