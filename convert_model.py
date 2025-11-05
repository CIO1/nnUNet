import os
import torch
import numpy as np
from typing import Tuple
import onnx

import pydoc
import warnings
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import load_json, join


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: bool = True):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    if nw_class is None:
        warnings.warn(f'网络类 {network_class} 未找到，尝试从dynamic_network_architectures中查找...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(
            join(dynamic_network_architectures.__path__[0], "architectures"),
            network_class.split(".")[-1],
            'dynamic_network_architectures.architectures'
        )
        if nw_class is None:
            raise ImportError('无法找到网络类，请检查plans文件是否正确')

    architecture_kwargs['deep_supervision'] = deep_supervision
    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


class MultiFormatExporter:
    def __init__(self, plans_path: str, config_name: str = "2d"):
        self.plans_dict = load_json(plans_path)
        self.config = self.plans_dict["configurations"][config_name]
        self.arch_class_name = self.config["architecture"]["network_class_name"]
        self.arch_kwargs = self.config["architecture"]["arch_kwargs"]
        self.arch_kwargs_req_import = self.config["architecture"]["_kw_requires_import"]

    def create_model(self, input_channels: int, output_channels: int,
                     model_weights_path: str, deep_supervision: bool = True) -> torch.nn.Module:
        """创建模型并加载权重（供导出使用）"""
        model = get_network_from_plans(
            arch_class_name=self.arch_class_name,
            arch_kwargs=self.arch_kwargs,
            arch_kwargs_req_import=self.arch_kwargs_req_import,
            input_channels=input_channels,
            output_channels=output_channels,
            deep_supervision=deep_supervision
        )
        checkpoint = torch.load(model_weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["network_weights"])
        #model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        return model

    def export_torchscript(self, model: torch.nn.Module, output_path: str,
                           input_shape: Tuple[int, int, int],
                           dynamic_input: bool = False) -> None:
        """导出为TorchScript格式（供libtorch/C++调用）"""

        # 包装模型，统一输出为单张量（处理深度监督）
        class WrappedModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                out = self.base_model(x)
                return out[-1] if isinstance(out, list) else out  # 取最后一层输出

        wrapped_model = WrappedModel(model)

        # 创建示例输入
        dummy_input = torch.randn(1, *input_shape)  # [1, C, H, W]

        # 追踪模型
        if dynamic_input:
            # 对于动态输入，使用脚本模式而非追踪模式
            script_model = torch.jit.script(wrapped_model)
            print("已导出支持动态输入尺寸的TorchScript模型")
        else:
            # 固定输入尺寸，使用追踪模式
            script_model = torch.jit.trace(wrapped_model, dummy_input)
            print("已导出固定输入尺寸的TorchScript模型")

        script_model.save(output_path)
        print(f"TorchScript模型（libtorch兼容）已保存：{output_path}")

    def export_onnx(self, model: torch.nn.Module, output_path: str,
                    input_shape: Tuple[int, int, int],
                    dynamic_input: bool = False) -> None:
        """导出为ONNX格式"""

        # 包装模型，确保输出为单张量（与TorchScript保持一致）
        class WrappedModelONNX(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                out = self.base_model(x)
                return out[-1] if isinstance(out, list) else out  # 取最后一层输出

        wrapped_model = WrappedModelONNX(model)
        dummy_input = torch.randn(1, *input_shape)  # [1, C, H, W]

        # 配置动态轴
        dynamic_axes = None
        if dynamic_input:
            # 完全动态：批次大小、高度和宽度都可以变化
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
            print("已导出支持动态输入尺寸的ONNX模型")
        else:
            # 仅批次大小动态
            dynamic_axes = None
            print("已导出固定维度动态的ONNX模型")

        # 导出ONNX
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


# 使用示例
if __name__ == "__main__":
    # 配置路径
    # PLANS_JSON_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_preprocessed\Dataset001_dx0904\nnUNetResEncUNetMPlans.json"
    # MODEL_WEIGHTS_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_results\Dataset001_dx0904\nnUNetTrainer__nnUNetResEncUNetMPlans__2d\fold_0\checkpoint_final.pth"
    PLANS_JSON_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_results\Dataset005_dx1024\nnUNetTrainer__nnUNetResEncUNetMPlans__2d\plans.json"
    MODEL_WEIGHTS_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_results\Dataset005_dx1024\nnUNetTrainer__nnUNetResEncUNetMPlans__2d\fold_0\checkpoint_final.pth"
    OUTPUT_TORCHSCRIPT = r"model_output\20251022\libtorch_model__fixInput384.pt"  # 供C++调用
    OUTPUT_ONNX = r"model_output\20251022\onnx_model_fixInput384.onnx"  # ONNX格式

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_TORCHSCRIPT), exist_ok=True)

    # 模型参数
    input_channels = 1
    output_channels = 3
    deep_supervision = False  # 若使用残差预置 训练时是有开启深度监督的，但是导出的时候我们一般只要最后一层 故这里设置为False
    input_shape = (1, 1536, 1536)  # (C, H, W)，输入图片尺寸
    dynamic_input = False  # 设置为True启用动态输入尺寸

    # 初始化导出器
    exporter = MultiFormatExporter(plans_path=PLANS_JSON_PATH)

    # 创建模型（共用一个模型实例）
    model = exporter.create_model(
        input_channels=input_channels,
        output_channels=output_channels,
        model_weights_path=MODEL_WEIGHTS_PATH,
        deep_supervision=deep_supervision
    )

    # 导出为TorchScript（供libtorch/C++）
    exporter.export_torchscript(model, OUTPUT_TORCHSCRIPT, input_shape,dynamic_input)

    # 导出为ONNX
    exporter.export_onnx(model, OUTPUT_ONNX, input_shape,dynamic_input)
