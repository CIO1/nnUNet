import os
import argparse


def rename_files(folder_path):
    """
    将文件夹中同名的PNG和JSON文件重命名为casexxx_0000格式

    Args:
        folder_path: 包含文件的文件夹路径
    """
    # 获取文件夹中所有文件
    all_files = os.listdir(folder_path)

    # 分离PNG和JSON文件
    png_files = [f for f in all_files if f.lower().endswith('.png')]
    json_files = [f for f in all_files if f.lower().endswith('.json')]

    # 提取文件名（不含扩展名）
    png_basenames = {os.path.splitext(f)[0] for f in png_files}
    json_basenames = {os.path.splitext(f)[0] for f in json_files}

    # 找到同时存在PNG和JSON的文件名
    common_basenames = sorted(png_basenames & json_basenames)

    if not common_basenames:
        print("没有找到同时存在PNG和JSON的同名文件")
        return

    print(f"找到 {len(common_basenames)} 对需要重命名的文件")

    # 遍历并按顺序重命名
    for i, basename in enumerate(common_basenames, 1):
        # 生成新的文件名，xxx从001开始
        new_basename = f"case{i:03d}_0000"

        # 原文件路径
        old_png_path = os.path.join(folder_path, f"{basename}.png")
        old_json_path = os.path.join(folder_path, f"{basename}.json")

        # 新文件路径
        new_png_path = os.path.join(folder_path, f"{new_basename}.png")
        new_json_path = os.path.join(folder_path, f"{new_basename}.json")

        # 重命名文件
        try:
            os.rename(old_png_path, new_png_path)
            os.rename(old_json_path, new_json_path)
            print(f"重命名成功: {basename} -> {new_basename}")
        except Exception as e:
            print(f"重命名失败 {basename}: {str(e)}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量重命名PNG和JSON文件')
    parser.add_argument('folder', help='包含文件的文件夹路径')
    args = parser.parse_args()

    # 检查文件夹是否存在
    if not os.path.isdir(args.folder):
        print(f"错误: {args.folder} 不是一个有效的文件夹路径")
    else:
        rename_files(args.folder)
