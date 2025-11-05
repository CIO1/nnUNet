import os
import numpy as np
import torch
import cv2
from typing import Tuple
from skimage import io
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero

# 导入模型创建工具函数
import pydoc
import warnings
from typing import Union
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # 尝试从dynamic_network_architectures中查找网络类
    if nw_class is None:
        warnings.warn(f'网络类 {network_class} 未找到，尝试从dynamic_network_architectures中查找...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(
            join(dynamic_network_architectures.__path__[0], "architectures"),
            network_class.split(".")[-1],
            'dynamic_network_architectures.architectures'
        )
        if nw_class is not None:
            print(f'找到网络类: {nw_class}')
        else:
            raise ImportError('无法找到网络类，请检查plans文件是否正确')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,** architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


class PredictPreprocessor:
    """适配NaturalImage2DIO的预测预处理类，支持单图和批量预测及结果可视化"""

    def __init__(self, plans_path: str, config_name: str = "2d"):
        # 加载JSON配置
        self.plans_dict = load_json(plans_path)
        self.config_name = config_name
        self.config = self.plans_dict["configurations"][config_name]

        # 提取关键参数
        self.target_spacing = self.config["spacing"]  # [1.0, 1.0]
        self.normalize_stats = self.plans_dict["foreground_intensity_properties_per_channel"]["0"]
        self.transpose_order = self.plans_dict["transpose_forward"]  # [0,1,2]
        self.image_reader = NaturalImage2DIO()  # 2D图像读取器

        # 网络配置（用于创建模型）
        self.arch_class_name = self.config["architecture"]["network_class_name"]
        self.arch_kwargs = self.config["architecture"]["arch_kwargs"]
        self.arch_kwargs_req_import = self.config["architecture"]["_kw_requires_import"]

        # 定义类别颜色映射（可根据实际类别数和需求调整）
        self.color_map = {
            0: (0, 0, 0),  # 背景-黑色（不参与融合）
            1: (0, 255, 0),  # 类别1-绿色
            2: (0, 0, 255),  # 类别2-红色
            # 如需更多类别，在此继续添加
        }

    def preprocess2(self, img_path: str) -> Tuple[torch.Tensor, dict]:
        # 1. 读取图像
        img_array, reader_metadata = self.image_reader.read_images([img_path])
        img_np = np.squeeze(img_array, axis=1)  # [C, 1, H, W] → [C, H, W]
        mean = img_np.mean()
        std = img_np.std()
        img_normalized = (img_np - mean) / (std + 1e-8)

        input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).float()
        metadata = {"img_path": img_path}

        return input_tensor, metadata

    def preprocess(self, img_path: str) -> Tuple[torch.Tensor, dict]:
        # 1. 读取图像
        img_array, reader_metadata = self.image_reader.read_images([img_path])
        img_np = np.squeeze(img_array, axis=1)  # [C, 1, H, W] → [C, H, W]

        # 2. 提取元数据
        original_spacing = list(reader_metadata["spacing"][1:])  # [1.0, 1.0]
        original_shape = img_np.shape[1:]  # [H, W]
        metadata = {
            "original_spacing": original_spacing,
            "original_shape": original_shape,
            "img_path": img_path,
            "full_spacing": reader_metadata["spacing"]
        }

        # 3. 维度转置
        img_transposed = img_np.transpose(self.transpose_order)

        # 4. 重采样
        target_shape = [
            int(round(original_shape[i] * original_spacing[i] / self.target_spacing[i]))
            for i in range(len(original_shape))
        ]
        img_4d = np.expand_dims(img_transposed, axis=-1)  # [C, H, W] → [C, H, W, 1]
        resampled_4d = resample_data_or_seg_to_shape(
            data=img_4d,
            new_shape=target_shape + [1],
            current_spacing=original_spacing + [1.0],
            new_spacing=self.target_spacing + [1.0],
            **self.config["resampling_fn_data_kwargs"]
        )
        img_resampled = np.squeeze(resampled_4d, axis=-1)  # [C, H, W, 1] → [C, H, W]

        # 5. Z-Score归一化
        mean = self.normalize_stats["mean"]
        std = self.normalize_stats["std"]
        img_normalized = (img_resampled - mean) / (std + 1e-8)

        # 6. 裁剪全0背景
        img_normalized_with_z = np.expand_dims(img_normalized, axis=1)  # 添加伪Z维度
        img_cropped_with_z, _, bbox_3d = crop_to_nonzero(img_normalized_with_z, None)

        # 校验边界框
        if len(bbox_3d) != 6:
            print(f"警告：图像 {os.path.basename(img_path)} 裁剪失败，使用原始尺寸")
            bbox_2d = (0, original_shape[0], 0, original_shape[1])
            img_cropped = img_normalized
        else:
            bbox_2d = (bbox_3d[2], bbox_3d[3], bbox_3d[4], bbox_3d[5])
            img_cropped = np.squeeze(img_cropped_with_z, axis=1)

        metadata["crop_bbox"] = bbox_2d

        # 7. 转为模型输入格式
        input_tensor = torch.from_numpy(img_cropped).unsqueeze(0).float()
        return input_tensor, metadata

    def create_model(self, input_channels: int, output_channels: int, deep_supervision: bool = True) -> torch.nn.Module:
        """根据plans配置创建模型"""
        return get_network_from_plans(
            arch_class_name=self.arch_class_name,
            arch_kwargs=self.arch_kwargs,
            arch_kwargs_req_import=self.arch_kwargs_req_import,
            input_channels=input_channels,
            output_channels=output_channels,
            allow_init=True,
            deep_supervision=deep_supervision
        )

    def postprocess_prediction(self, pred: torch.Tensor, metadata: dict, output_dir: str) -> None:
        """后处理预测结果，生成掩模并与原图叠加（仅前景区域融合）"""
        # 处理深度监督输出（取最后一层输出）
        if isinstance(pred, list):
            pred = pred[0]  # 深度监督通常最后一层是最佳结果

        # 1. 处理预测结果
        pred_np = pred.squeeze(0).cpu().numpy()  # [num_classes, H, W]
        pred_argmax = np.argmax(pred_np, axis=0).astype(np.uint8)  # [H, W]

        # 2. 读取原图
        img_path = metadata["img_path"]
        original_img = cv2.imread(img_path)  # 读取为BGR格式
        if original_img is None:
            raise FileNotFoundError(f"无法读取原图：{img_path}")

        # 确保原图是三通道（如果是灰度图则转换）
        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

        # 3. 调整掩模尺寸以匹配原图
        orig_h, orig_w = original_img.shape[:2]
        pred_h, pred_w = pred_argmax.shape
        if (pred_h, pred_w) != (orig_h, orig_w):
            # 调整掩模尺寸到原图大小
            pred_argmax = cv2.resize(pred_argmax, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 4. 将掩模转换为彩色（仅前景着色）
        colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for class_id, color in self.color_map.items():
            if class_id != 0:  # 跳过背景
                colored_mask[pred_argmax == class_id] = color

        # 5. 创建前景掩码（背景为0，前景为1）
        foreground_mask = (pred_argmax != 0).astype(np.uint8)  # [H, W]
        foreground_mask_3d = np.stack([foreground_mask] * 3, axis=-1)  # 扩展为3通道 [H, W, 3]

        # 6. 仅在前景区域叠加掩模（背景保持原图）
        alpha = 0.8  # 原图权重
        beta = 0.2   # 掩模权重
        # 计算前景区域的融合结果
        foreground_overlay = cv2.addWeighted(
            original_img * foreground_mask_3d,  # 仅取原图前景区域
            alpha,
            colored_mask,  # 掩模（只有前景有颜色）
            beta,
            0
        )
        # 背景区域保持原图，前景区域使用融合结果
        overlay_img = (original_img * (1 - foreground_mask_3d)) + foreground_overlay

        # 7. 准备保存路径
        img_basename = os.path.basename(img_path)
        img_name = os.path.splitext(img_basename)[0]
        mask_save_path = os.path.join(output_dir, f"{img_name}_mask.png")
        overlay_save_path = os.path.join(output_dir, f"{img_name}_overlay.png")

        # 8. 保存结果
        cv2.imwrite(mask_save_path, pred_argmax*122)
        cv2.imwrite(overlay_save_path, overlay_img)
        print(f"掩模保存至：{mask_save_path}")
        print(f"叠加图像保存至：{overlay_save_path}")

    def predict_single_image(self, img_path: str, model: torch.nn.Module, device: str, output_dir: str) -> None:
        """预测单张图片并保存结果"""
        if not os.path.exists(img_path):
            print(f"警告：图片 {img_path} 不存在，跳过")
            return

        # 预处理
        input_tensor, metadata = self.preprocess2(img_path)

        # 预测
        model.eval()
        with torch.no_grad():
            pred = model(input_tensor.to(device))

        # 后处理并保存
        self.postprocess_prediction(pred, metadata, output_dir)

    def predict_folder(self, input_folder: str, model: torch.nn.Module, device: str, output_dir: str) -> None:
        """批量预测文件夹中所有图片"""
        # 确保输出文件夹存在
        os.makedirs(output_dir, exist_ok=True)

        # 支持的图像格式
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        # 遍历文件夹处理图片
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(image_extensions):
                img_path = os.path.join(input_folder, filename)
                print(f"\n处理图像：{img_path}")
                try:
                    self.predict_single_image(img_path, model, device, output_dir)
                except Exception as e:
                    print(f"处理 {filename} 失败：{str(e)}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    PLANS_JSON_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_results\Dataset004_dx1020\nnUNetTrainer__nnUNetResEncUNetMPlans__2d\plans.json"
    #INPUT_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_raw\Dataset003_dx1020\imagesTr"  # 可改为单张图片路径或文件夹路径
    INPUT_PATH = r"F:\ImageDataSet\RawData\output_tif"
    MODEL_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_results\Dataset004_dx1020\nnUNetTrainer__nnUNetResEncUNetMPlans__2d\fold_0\checkpoint_final.pth"
    OUTPUT_DIR = r"F:\ImageDataSet\RawData\output_tif\out"  # 结果保存文件夹

    # 初始化预处理器
    preprocessor = PredictPreprocessor(plans_path=PLANS_JSON_PATH)

    # 创建模型
    input_channels = 1  # 单通道输入
    output_channels = 3  # 实际类别数（需与训练时一致）
    deep_supervision = True
    model = preprocessor.create_model(input_channels, output_channels, deep_supervision)

    # 加载模型权重
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    assert os.path.exists(MODEL_PATH), f"模型文件不存在：{MODEL_PATH}"

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print("Checkpoint 包含的键名：")
    for key in checkpoint.keys():
        print(f"- {key}")
    model_weights = checkpoint["network_weights"]
    model.load_state_dict(model_weights)
    model = model.to(device)
    model.eval()
    print("模型加载完成并设置为评估模式")

    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 根据输入类型选择处理方式
    if os.path.isfile(INPUT_PATH):
        # 处理单张图片
        print(f"开始处理单张图片：{INPUT_PATH}")
        preprocessor.predict_single_image(INPUT_PATH, model, device, OUTPUT_DIR)
    elif os.path.isdir(INPUT_PATH):
        # 处理文件夹
        print(f"开始处理文件夹：{INPUT_PATH}")
        preprocessor.predict_folder(INPUT_PATH, model, device, OUTPUT_DIR)
    else:
        print(f"错误：输入路径不存在 - {INPUT_PATH}")