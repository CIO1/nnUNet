import onnxruntime as ort
import numpy as np
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO

image_reader = NaturalImage2DIO()
def load_onnx_model(model_path):
    """加载ONNX模型并返回推理会话"""
    try:
        # 创建ONNX Runtime推理会话
        session = ort.InferenceSession(model_path)
        print("模型加载成功")
        return session
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def get_input_output_names(session):
    """获取模型的输入和输出名称"""
    input_names = [input_node.name for input_node in session.get_inputs()]
    output_names = [output_node.name for output_node in session.get_outputs()]
    return input_names, output_names



def preprocess_input(img_path, input_shape=None):
    """
    预处理输入数据以适应模型要求
    这里只是示例，实际预处理应根据模型要求进行调整
    """
        # 1. 读取图像
    img_array, reader_metadata = image_reader.read_images([img_path])
    img_np = np.squeeze(img_array, axis=1)  # [C, 1, H, W] → [C, H, W]
    mean = img_np.mean()
    std = img_np.std()
    mean1 =16631.0117187
    std1 = 12366.90234375
    img_normalized1 = (img_np - mean1) / (std1 + 1e-8)
    img_normalized1 = np.expand_dims(img_normalized1, axis=0)

    return img_normalized1


def run_inference(session, input_data):
    """使用ONNX模型进行推理"""
    if not session:
        print("未加载有效的模型")
        return None

    try:
        # 获取输入输出名称
        input_names, output_names = get_input_output_names(session)

        # 确保输入数据是字典格式
        if isinstance(input_data, dict):
            inputs = input_data
        else:
            # 如果输入是单一数组，使用第一个输入名称
            inputs = {input_names[0]: input_data}

        # 执行推理
        outputs = session.run(output_names, inputs)

        return outputs
    except Exception as e:
        print(f"推理过程出错: {e}")
        return None


def postprocess_output(outputs):
    """
    后处理模型输出
    这里只是示例，实际后处理应根据模型和任务要求进行调整
    """
    # 对于分类任务，可能需要取softmax或找到最大概率的类别
    # 对于分割任务，可能需要处理掩码等

    # 示例：如果是分类任务，返回每个样本的预测类别
    np.squeeze(outputs,0)
    if len(outputs) > 0:
        # 假设第一个输出是预测分数
        predictions = np.argmax(outputs[0], axis=1)
        return predictions
    return None


def main():
    # 模型路径
    model_path = r"F:\CJY\deep-learning\nnUNet\model_output\onnx_model.onnx"  # 替换为你的ONNX模型路径

    # 加载模型
    session = load_onnx_model(model_path)
    if not session:
        return

    # 打印输入输出信息
    input_names, output_names = get_input_output_names(session)
    print(f"输入名称: {input_names}")
    print(f"输出名称: {output_names}")

    # 获取输入形状信息
    input_shapes = [session.get_inputs()[i].shape for i in range(len(input_names))]
    print(f"输入形状: {input_shapes}")

    # 创建示例输入数据（这里使用随机数据作为示例）

    PREDICT_IMG_PATH = r"F:\CJY\deep-learning\pytorch-CycleGAN-and-pix2pi\imgDataset\checkedDATA0904\png\case104_0000.png"

    # 预处理输入
    processed_input = preprocess_input(PREDICT_IMG_PATH)

    # 执行推理
    outputs = run_inference(session, processed_input)

    if outputs:
        print("推理结果:")
        for i, output in enumerate(outputs):
            print(f"输出 {i + 1} 形状: {output.shape}")

        # 后处理输出（根据实际任务调整）
        results = postprocess_output(output)
        if results is not None:
            print(f"后处理结果: {results}")


if __name__ == "__main__":
    main()
