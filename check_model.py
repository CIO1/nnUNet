import onnx
from onnx import checker

# 加载 ONNX 模型
model = onnx.load(r"F:\CJY\deep-learning\nnUNet\model_output\onnx_model.onnx")

# 检查模型一致性
checker.check_model(model)

print("模型检查通过，结构正确！")