import os
import torch
import numpy as np
from typing import Tuple
import onnx

import warnings
from batchgenerators.utilities.file_and_folder_operations import load_json, join

# 导入新模型所需的依赖类
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
import torch.nn as nn

# 导入UMambaBot类（根据实际路径调整）
from nnunetv2.nets.UMambaBot_2d import UMambaBot


# ------------------------------
# 关键修改1：配置设备（优先使用CUDA）
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise RuntimeError("Mamba算子不支持CPU，需使用带CUDA的GPU环境运行")


def get_umamba_bot_2d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    """新模型结构：构建UMambaBot网络"""
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaBot'
    network_class = UMambaBot
    kwargs = {
        'UMambaBot': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,** conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    return model


class MultiFormatExporter:
    def __init__(self, plans_path: str, config_name: str = "2d"):
        self.plans_path = plans_path
        self.plans_dict = load_json(plans_path)
        self.config = self.plans_dict["configurations"][config_name]

        # 加载dataset.json（与plans.json同目录）
        dataset_json_path = join(os.path.dirname(plans_path), "dataset.json")
        self.dataset_json = load_json(dataset_json_path)

        # 初始化计划管理器和配置管理器
        self.plans_manager = PlansManager(self.plans_dict)
        self.configuration_manager = ConfigurationManager(self.config)

    def create_model(self, input_channels: int,
                     model_weights_path: str,
                     deep_supervision: bool = True) -> torch.nn.Module:
        """创建UMambaBot模型并加载权重（确保在CUDA上）"""
        # 构建模型
        model = get_umamba_bot_2d_from_plans(
            plans_manager=self.plans_manager,
            dataset_json=self.dataset_json,
            configuration_manager=self.configuration_manager,
            num_input_channels=input_channels,
            deep_supervision=deep_supervision
        )

        # ------------------------------
        # 关键修改2：权重加载到CUDA，模型移到CUDA
        # ------------------------------
        checkpoint = torch.load(model_weights_path, map_location=device)  # 修正：用device而非"GPU"
        model.load_state_dict(checkpoint["network_weights"])
        model = model.to(device)  # 模型移到CUDA
        model.eval()
        return model

    def export_torchscript(self, model: torch.nn.Module, output_path: str,
                           input_shape: Tuple[int, int, int],
                           dynamic_input: bool = False) -> None:
        """导出为TorchScript格式（确保输入在CUDA上）"""

        class WrappedModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            @torch.jit.ignore
            def forward(self, x):
                out = self.base_model(x)
                return out[-1] if isinstance(out, list) else out

        wrapped_model = WrappedModel(model)
        # ------------------------------
        # 关键修改3：虚拟输入移到CUDA
        # ------------------------------
        dummy_input = torch.randn(1, *input_shape, device=device)  # 添加device=device

        if dynamic_input:
            script_model = torch.jit.script(wrapped_model)
            print("已导出支持动态输入尺寸的TorchScript模型")
        else:
            script_model = torch.jit.trace(wrapped_model, dummy_input)
            print("已导出固定输入尺寸的TorchScript模型")

        script_model.save(output_path)
        print(f"TorchScript模型已保存：{output_path}")

    def export_onnx(self, model: torch.nn.Module, output_path: str,
                    input_shape: Tuple[int, int, int],
                    dynamic_input: bool = False) -> None:
        """导出为ONNX格式（确保输入在CUDA上）"""

        class WrappedModelONNX(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                out = self.base_model(x)
                return out[-1] if isinstance(out, list) else out

        wrapped_model = WrappedModelONNX(model)
        # ------------------------------
        # 关键修改4：虚拟输入移到CUDA
        # ------------------------------
        dummy_input = torch.randn(1, *input_shape, device=device)  # 添加device=device

        dynamic_axes = None
        if dynamic_input:
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
            print("已导出支持动态输入尺寸的ONNX模型")
        else:
            print("已导出固定维度的ONNX模型")

        # 导出ONNX（注意：ONNX导出时会自动处理设备无关性）
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )

        # 验证ONNX模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX模型已保存并验证：{output_path}")


if __name__ == "__main__":
    # 配置路径（修正：Linux系统使用正斜杠/）
    PLANS_JSON_PATH = r"../data/nnUNet_results/Dataset005_dx1024/nnUNetTrainerUMambaBot__nnUNetPlans__2d/plans.json"
    MODEL_WEIGHTS_PATH = r"../data/nnUNet_results/Dataset005_dx1024/nnUNetTrainerUMambaBot__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
    # ------------------------------
    # 关键修改5：路径使用正斜杠（适应Linux）
    # ------------------------------
    OUTPUT_TORCHSCRIPT = r"../model_output/20251022/libtorch_model__fixInput.pt"
    OUTPUT_ONNX = r"../model_output/20251022/onnx_model_fixInput.onnx"

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_TORCHSCRIPT), exist_ok=True)

    # 模型参数
    input_channels = 1
    deep_supervision = False
    input_shape = (1, 1536, 1536)  # (C, H, W)
    dynamic_input = False

    # 初始化导出器并创建模型
    exporter = MultiFormatExporter(plans_path=PLANS_JSON_PATH)
    model = exporter.create_model(
        input_channels=input_channels,
        model_weights_path=MODEL_WEIGHTS_PATH,
        deep_supervision=deep_supervision
    )

    # 导出模型

    #exporter.export_onnx(model, OUTPUT_ONNX, input_shape, dynamic_input)
    exporter.export_torchscript(model, OUTPUT_TORCHSCRIPT, input_shape, dynamic_input)
