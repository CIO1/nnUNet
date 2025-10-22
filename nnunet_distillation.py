import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, Union, List
from tqdm import tqdm
from skimage import io
from batchgenerators.utilities.file_and_folder_operations import load_json, join,save_json
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.custom_dice_metrics import SoftDiceLoss,get_tp_fp_fn_tn
# 导入模型创建工具函数
import pydoc
import warnings
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    """从plans配置创建网络模型"""
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
        num_classes=output_channels, **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network

def save_checkpoint(model: torch.nn.Module, path: str, optimizer: optim.Optimizer, epoch: int, loss: float):
    """保存模型检查点（补充实现）"""
    torch.save({
        "current_epoch": epoch,
        "network_weights": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }, path)
    print(f"检查点已保存至: {path}")

class DistillationDataset(Dataset):
    """蒸馏训练数据集"""

    def __init__(self, image_paths: List[str], seg_paths: List[str]):
        self.image_paths = image_paths
        self.seg_paths = seg_paths
        self.image_reader = NaturalImage2DIO()  # 2D图像读取器
        assert len(image_paths) == len(seg_paths), "图像和标签数量必须一致"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_array, _ = self.image_reader.read_images([self.image_paths[idx]])
        img_np = np.squeeze(img_array, axis=1)  # [C, 1, H, W] → [C, H, W]
        mean = img_np.mean()
        std = img_np.std()
        img_normalized = (img_np - mean) / (std + 1e-8)

        image = torch.from_numpy(img_normalized).float()

        seg_array,_=self.image_reader.read_images([self.seg_paths[idx]])
        seg_np = np.squeeze(seg_array, axis=(0, 1))  # [C, 1, H, W] → [ H, W]
        seg = torch.from_numpy(seg_np).long()


        return {
            "image":image,
            "seg":seg,
            "image_path": self.image_paths[idx],
            "seg_path": self.seg_paths[idx]
        }


class DistillationTrainer:
    """知识蒸馏训练器（适配nnUNet模型+16bit单通道数据）"""

    def __init__(self,
                 teacher_model: torch.nn.Module,
                 student_model: torch.nn.Module,
                 num_classes: int,
                 output_dir: str,
                 device: torch.device,
                 temperature: float = 3.0,
                 alpha: float = 0.7,
                 learning_rate: float = 1e-4,
                 batch_size: int = 8,
                 num_epochs: int = 100,
                 deep_supervision: bool = False):
        """
        初始化参数
        Args:
            num_classes: 类别数（必须与模型输出匹配）
            deep_supervision: 是否启用深度监督（需与模型输出格式一致）
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.device = device
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重（(1-alpha)为分割损失权重）
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.deep_supervision = deep_supervision

        # -------------------------- 初始化基础组件 --------------------------
        self._init_output_dir()
        self._init_loss_functions()
        self._init_optimizer_scheduler()
        self._init_train_history()

        # 教师模型固定（不参与训练）
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _init_output_dir(self):
        """创建输出目录（checkpoints+日志）"""
        self.checkpoint_dir = join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"输出目录初始化完成: {self.output_dir}")

    def _init_loss_functions(self):
        """初始化损失函数（分割损失+蒸馏损失）"""
        # 1. 分割损失：SoftDiceLoss（nnUNet常用，支持多类别）
        self.seg_loss_fn = SoftDiceLoss(
            apply_nonlin=lambda x: torch.softmax(x, dim=1),  # 关键修改：显式指定dim=1
            batch_dice=True,
            do_bg=False  # 根据需求决定是否包含背景
        )

        # 2. 蒸馏损失：KL散度（衡量概率分布差异）
        self.distill_loss_fn = torch.nn.KLDivLoss(
            reduction="batchmean",  # 批次平均
            log_target=False  # 输入为log_softmax，目标为softmax
        )

    def _init_optimizer_scheduler(self):
        """初始化优化器和学习率调度器"""
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5  # L2正则化（防止过拟合）
        )

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # 基于验证损失最小化调整
            factor=0.2,  # 学习率衰减因子
            patience=10,  # 10个epoch无提升则衰减
            verbose=True,  # 打印调度日志
            min_lr=1e-7  # 最小学习率（避免衰减至0）
        )

    def _init_train_history(self):
        """初始化训练历史记录（损失+指标）"""
        self.train_history = {
            "train_total_loss": [],
            "train_seg_loss": [],
            "train_distill_loss": [],
            "train_dice": [],
            "val_total_loss": [],
            "val_seg_loss": [],
            "val_distill_loss": [],
            "val_dice": []
        }

    def _convert_label_to_onehot(self, label: torch.Tensor) -> torch.Tensor:
        """将类别索引标签转换为one-hot格式（适配Dice损失）
        Args:
            label: [B, H, W] (long)
        Returns:
            onehot_label: [B, C, H, W] (float)
        """
        batch_size, h, w = label.shape
        onehot = torch.zeros(batch_size, self.num_classes, h, w, device=self.device)
        return onehot.scatter_(1, label.unsqueeze(1), 1.0)  # 在通道维度散射

    def compute_distillation_loss(self, student_output: Union[torch.Tensor, List[torch.Tensor]],
                                  teacher_output: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """计算蒸馏损失（支持深度监督）"""
        # 深度监督：模型输出为列表（多尺度特征图）
        if self.deep_supervision and isinstance(student_output, list) and isinstance(teacher_output, list):
            assert len(student_output) == len(teacher_output), "深度监督输出层数不匹配"
            total_distill_loss = 0.0
            # 深层特征图权重更高（nnUNet深度监督策略）
            weights = [0.5 ** (len(student_output) - i - 1) for i in range(len(student_output))]

            for s_out, t_out, weight in zip(student_output, teacher_output, weights):
                # 教师输出→概率（不计算梯度）
                t_prob = torch.softmax(t_out / self.temperature, dim=1)
                # 学生输出→log概率
                s_log_prob = torch.log_softmax(s_out / self.temperature, dim=1)
                # 计算KL散度（乘温度平方还原损失量级）
                total_distill_loss += weight * self.distill_loss_fn(s_log_prob, t_prob) * (self.temperature ** 2)

            return total_distill_loss / sum(weights)  # 权重归一化

        # 普通模式：模型输出为单张量
        else:
            t_prob = torch.softmax(teacher_output / self.temperature, dim=1)
            s_log_prob = torch.log_softmax(student_output / self.temperature, dim=1)
            return self.distill_loss_fn(s_log_prob, t_prob) * (self.temperature ** 2)

    def compute_segmentation_loss(self, model_output: Union[torch.Tensor, List[torch.Tensor]],
                                  label: torch.Tensor) -> torch.Tensor:
        """计算分割损失（支持深度监督+one-hot标签）"""
        # 标签转换为one-hot（Dice损失需求）
        onehot_label = self._convert_label_to_onehot(label).to(self.device)

        # 深度监督模式
        if self.deep_supervision and isinstance(model_output, list):
            total_seg_loss = 0.0
            weights = [0.5 ** (len(model_output) - i - 1) for i in range(len(model_output))]

            for out, weight in zip(model_output, weights):
                # 确保特征图尺寸与标签匹配（深度监督可能输出小尺寸特征图）
                if out.shape[2:] != onehot_label.shape[2:]:
                    out = torch.nn.functional.interpolate(
                        out, size=onehot_label.shape[2:], mode="bilinear", align_corners=False
                    )
                total_seg_loss += weight * self.seg_loss_fn(out, onehot_label)

            return total_seg_loss / sum(weights)

        # 普通模式
        else:
            return self.seg_loss_fn(model_output, onehot_label)

    def compute_dice_score(self, model_output: Union[torch.Tensor, List[torch.Tensor]],
                           label: torch.Tensor) -> float:
        """计算Dice评分（排除背景，适配多类别）"""
        # 处理深度监督：取最后一层（最大尺寸）输出
        if self.deep_supervision and isinstance(model_output, list):
            model_output = model_output[-1]  # 最后一层输出尺寸与输入一致

        # 模型输出→概率→预测类别
        output_prob = torch.softmax(model_output, dim=1)
        output_pred = torch.argmax(output_prob, dim=1)  # [B, H, W]

        # 计算TP/FP/FN（排除背景类：label=0）
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            output_pred, label,
            num_classes=self.num_classes,
            include_background=False  # 排除背景计算Dice
        )

        # 计算多类别Dice均值（避免除以0）
        dice_per_class = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        mean_dice = dice_per_class.mean().item()
        return mean_dice

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练单个epoch"""
        self.student_model.train()
        self.teacher_model.eval()  # 教师模型仅用于推理

        # 记录当前epoch的损失和指标
        epoch_metrics = {
            "total_loss": [],
            "seg_loss": [],
            "distill_loss": [],
            "dice": []
        }

        progress_bar = tqdm(dataloader, desc="Training Epoch")
        for batch in progress_bar:
            # 1. 数据加载到设备
            images = batch["image"].to(self.device)  # [B, 1, H, W]
            labels = batch["seg"].to(self.device)  # [B, H, W] (long)

            # 2. 梯度清零
            self.optimizer.zero_grad()

            # 3. 前向传播
            with torch.no_grad():  # 教师模型不计算梯度
                teacher_output = self.teacher_model(images)
            student_output = self.student_model(images)

            # 4. 损失计算
            seg_loss = self.compute_segmentation_loss(student_output, labels)
            distill_loss = self.compute_distillation_loss(student_output, teacher_output)
            total_loss = (1 - self.alpha) * seg_loss + self.alpha * distill_loss

            # 5. 反向传播与优化
            total_loss.backward()
            self.optimizer.step()

            # 6. 指标计算
            dice_score = self.compute_dice_score(student_output, labels)

            # 7. 记录数据
            epoch_metrics["total_loss"].append(total_loss.item())
            epoch_metrics["seg_loss"].append(seg_loss.item())
            epoch_metrics["distill_loss"].append(distill_loss.item())
            epoch_metrics["dice"].append(dice_score)

            # 8. 更新进度条
            progress_bar.set_postfix({
                "tot_loss": f"{np.mean(epoch_metrics['total_loss']):.4f}",
                "seg_loss": f"{np.mean(epoch_metrics['seg_loss']):.4f}",
                "distill_loss": f"{np.mean(epoch_metrics['distill_loss']):.4f}",
                "dice": f"{np.mean(epoch_metrics['dice']):.4f}"
            })

        # 计算epoch均值
        epoch_avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        # 更新训练历史
        self.train_history["train_total_loss"].append(epoch_avg["total_loss"])
        self.train_history["train_seg_loss"].append(epoch_avg["seg_loss"])
        self.train_history["train_distill_loss"].append(epoch_avg["distill_loss"])
        self.train_history["train_dice"].append(epoch_avg["dice"])

        return epoch_avg

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证单个epoch（无梯度计算）"""
        self.student_model.eval()
        self.teacher_model.eval()

        epoch_metrics = {
            "total_loss": [],
            "seg_loss": [],
            "distill_loss": [],
            "dice": []
        }

        progress_bar = tqdm(dataloader, desc="Validation Epoch")
        for batch in progress_bar:
            # 数据加载
            images = batch["image"].to(self.device)
            labels = batch["seg"].to(self.device)

            # 前向传播
            teacher_output = self.teacher_model(images)
            student_output = self.student_model(images)

            # 损失计算
            seg_loss = self.compute_segmentation_loss(student_output, labels)
            distill_loss = self.compute_distillation_loss(student_output, teacher_output)
            total_loss = (1 - self.alpha) * seg_loss + self.alpha * distill_loss

            # 指标计算
            dice_score = self.compute_dice_score(student_output, labels)

            # 记录数据
            epoch_metrics["total_loss"].append(total_loss.item())
            epoch_metrics["seg_loss"].append(seg_loss.item())
            epoch_metrics["distill_loss"].append(distill_loss.item())
            epoch_metrics["dice"].append(dice_score)

            # 更新进度条
            progress_bar.set_postfix({
                "val_tot_loss": f"{np.mean(epoch_metrics['total_loss']):.4f}",
                "val_dice": f"{np.mean(epoch_metrics['dice']):.4f}"
            })

        # 计算均值并更新历史
        epoch_avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        self.train_history["val_total_loss"].append(epoch_avg["total_loss"])
        self.train_history["val_seg_loss"].append(epoch_avg["seg_loss"])
        self.train_history["val_distill_loss"].append(epoch_avg["distill_loss"])
        self.train_history["val_dice"].append(epoch_avg["dice"])

        return epoch_avg

    def save_train_history(self):
        """保存训练历史（JSON格式，便于后续分析）"""
        history_path = join(self.output_dir, "train_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            save_json(self.train_history, f)
        print(f"训练历史已保存至: {history_path}")

    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """完整训练流程（含验证+模型保存）"""
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True  # 加速GPU数据传输
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) if val_dataset is not None else None

        # 初始化最佳模型指标（基于验证Dice）
        best_val_dice = 0.0

        # 开始训练循环
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.num_epochs} ===")

            # 1. 训练epoch
            train_metrics = self.train_epoch(train_loader)
            print(f"训练结果 - 总损失: {train_metrics['total_loss']:.4f}, Dice: {train_metrics['dice']:.4f}")

            # 2. 验证epoch（如有验证集）
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                print(f"验证结果 - 总损失: {val_metrics['total_loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
                # 基于验证损失调整学习率
                self.lr_scheduler.step(val_metrics["total_loss"])
            else:
                # 无验证集时基于训练损失调整
                self.lr_scheduler.step(train_metrics["total_loss"])

            # 3. 保存普通检查点（每5个epoch保存一次，避免冗余）
            if epoch % 5 == 0:
                checkpoint_path = join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(
                    model=self.student_model,
                    path=checkpoint_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=train_metrics["total_loss"]
                )

            # 4. 保存最佳模型（基于验证Dice）
            current_val_dice = val_metrics["dice"] if val_metrics is not None else train_metrics["dice"]
            if current_val_dice > best_val_dice:
                best_val_dice = current_val_dice
                best_checkpoint_path = join(self.checkpoint_dir, "checkpoint_best.pth")
                save_checkpoint(
                    model=self.student_model,
                    path=best_checkpoint_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=current_val_dice  # 用Dice作为最佳指标
                )

        # 训练结束：保存最终模型和历史
        final_checkpoint_path = join(self.checkpoint_dir, "checkpoint_final.pth")
        save_checkpoint(
            model=self.student_model,
            path=final_checkpoint_path,
            optimizer=self.optimizer,
            epoch=self.num_epochs,
            loss=self.train_history["train_total_loss"][-1]
        )

        self.save_train_history()
        print(f"\n训练完成！最佳验证Dice: {best_val_dice:.4f}")


class DistillationModel:
    """用于知识蒸馏的预处理类，确保教师和学生模型使用一致的预处理流程"""
    def __init__(self,
                 teacher_plans_path: str,
                 student_plans_path: str,
                 teacher_model_path: str,
                 num_channels:int,
                 num_classes:int,
                 teacher_config_name: str = "2d",
                 student_config_name: str = "2d"):
        """
        初始化蒸馏预处理工具
        Args:
            teacher_plans_path: 教师模型的plans文件路径
            student_plans_path: 学生模型的plans文件路径
            teacher_model_path: 教师模型权重路径
            teacher_config_name: 教师模型配置名称
            student_config_name: 学生模型配置名称
        """
        # 加载教师和学生的配置
        self.teacher_plans = load_json(teacher_plans_path)
        self.student_plans = load_json(student_plans_path)
        self.teacher_model_path = teacher_model_path

        self.teacher_config = self.teacher_plans["configurations"][teacher_config_name]
        self.student_config = self.student_plans["configurations"][student_config_name]

        # 提取公共参数
        self.target_spacing = self.teacher_config["spacing"]  # 使用教师模型的间距作为基准
        self.normalize_stats = self.teacher_plans["foreground_intensity_properties_per_channel"]["0"]
        self.transpose_order = self.teacher_plans["transpose_forward"]
        self.num_channels=num_channels
        self.num_classes=num_classes
        # 网络配置
        self.teacher_arch = {
            "class_name": self.teacher_config["architecture"]["network_class_name"],
            "kwargs": self.teacher_config["architecture"]["arch_kwargs"],
            "req_import": self.teacher_config["architecture"]["_kw_requires_import"]
        }

        self.student_arch = {
            "class_name": self.student_config["architecture"]["network_class_name"],
            "kwargs": self.student_config["architecture"]["arch_kwargs"],
            "req_import": self.student_config["architecture"]["_kw_requires_import"]
        }

    def create_teacher_model(self,device, deep_supervision: bool = True) -> torch.nn.Module:
        """创建教师模型"""
        model=get_network_from_plans(
            arch_class_name=self.teacher_arch["class_name"],
            arch_kwargs=self.teacher_arch["kwargs"],
            arch_kwargs_req_import=self.teacher_arch["req_import"],
            input_channels=self.num_channels,
            output_channels=self.num_classes,
            allow_init=True,
            deep_supervision=deep_supervision
        )
        # 加载完整检查点，提取模型权重
        checkpoint = torch.load(self.teacher_model_path, map_location=device)
        model_weights = checkpoint["network_weights"]  # 从检查点中提取模型权重
        model.load_state_dict(model_weights)
        model = model.to(device)
        model.eval()
        return  model

    def create_student_model(self,device, deep_supervision: bool = True) -> torch.nn.Module:
        """创建学生模型"""
        model=get_network_from_plans(
            arch_class_name=self.student_arch["class_name"],
            arch_kwargs=self.student_arch["kwargs"],
            arch_kwargs_req_import=self.student_arch["req_import"],
            input_channels=self.num_channels,
            output_channels=self.num_classes,
            allow_init=True,
            deep_supervision=deep_supervision
        )
        model.to(device)
        return  model







# 使用示例
if __name__ == "__main__":
    # 配置路径
    TEACHER_PLANS_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_preprocessed\Dataset001_dx0904\nnUNetResEncUNetMPlans.json"
    STUDENT_PLANS_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_preprocessed\Dataset001_dx0904\nnUNetResEncUNetStudentPlans.json"
    TEACHER_MODEL_PATH = r"F:\CJY\deep-learning\nnUNet\nnUNet_results\Dataset001_dx0904\nnUNetTrainer__nnUNetResEncUNetMPlans__2d\fold_0\checkpoint_final.pth"
    OUTPUT_DIR = r"F:\CJY\deep-learning\nnUNet\nnUNet_results\distillation"

    # 训练数据路径（实际使用时替换为你的数据集路径列表）
    import glob

    train_image_paths = glob.glob(
        r"F:\CJY\deep-learning\nnUNet\nnUNet_raw\Dataset001_dx0904\imagesTr\*.png")[:80]
    train_seg_paths = [p.replace("imagesTr", "labelsTr").replace("_0000.png", ".png") for p in train_image_paths]

    # 验证数据路径
    val_image_paths = glob.glob(
        r"F:\CJY\deep-learning\nnUNet\nnUNet_raw\Dataset001_dx0904\imagesTr\*.png")[80:]

    val_seg_paths = [p.replace("imagesTr", "labelsTr").replace("_0000.png", ".png") for p in val_image_paths]
    # 创建模型（根据plans自动匹配网络结构）
    input_channels = 1  # 单通道输入
    output_channels = 3  # 替换为你的实际类别数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_supervision=False

    # 初始化蒸馏预处理器
    distillationModel = DistillationModel(
        teacher_plans_path=TEACHER_PLANS_PATH,
        student_plans_path=STUDENT_PLANS_PATH,
        teacher_model_path=TEACHER_MODEL_PATH,
        num_channels=input_channels,
        num_classes=output_channels
    )
    teacher_model=distillationModel.create_teacher_model(device=device,deep_supervision=deep_supervision)
    student_model=distillationModel.create_student_model(device=device,deep_supervision=deep_supervision)

    # 创建数据集
    train_dataset = DistillationDataset(train_image_paths, train_seg_paths)
    val_dataset = DistillationDataset(val_image_paths, val_seg_paths)

    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化蒸馏训练器
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        num_classes=output_channels,
        output_dir=OUTPUT_DIR,
        device=device,
        temperature=3.0,
        alpha=0.7,
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=50,
        deep_supervision=True
    )

    # 开始训练
    trainer.train(train_dataset, val_dataset)

