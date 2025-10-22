import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_tp_fp_fn_tn(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        include_background: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算真阳性(TP)、假阳性(FP)、假阴性(FN)和真阴性(TN)
    Args:
        predictions: 模型预测的类别索引，形状为 [B, H, W]
        targets: 真实标签，形状为 [B, H, W]
        num_classes: 类别总数
        include_background: 是否包含背景类(0)
    Returns:
        tp, fp, fn, tn: 每个类别的统计值，形状为 [num_classes]
    """
    # 初始化统计数组
    tp = torch.zeros(num_classes, device=predictions.device)
    fp = torch.zeros(num_classes, device=predictions.device)
    fn = torch.zeros(num_classes, device=predictions.device)
    tn = torch.zeros(num_classes, device=predictions.device)

    # 遍历每个类别计算混淆矩阵
    for cls in range(num_classes):
        # 跳过背景类（如果需要）
        if not include_background and cls == 0:
            continue

        # 预测为当前类且标签为当前类 → TP
        tp_cls = ((predictions == cls) & (targets == cls)).sum()
        # 预测为当前类但标签不是当前类 → FP
        fp_cls = ((predictions == cls) & (targets != cls)).sum()
        # 标签为当前类但预测不是当前类 → FN
        fn_cls = ((predictions != cls) & (targets == cls)).sum()
        # 标签不是当前类且预测不是当前类 → TN
        tn_cls = ((predictions != cls) & (targets != cls)).sum()

        tp[cls] = tp_cls
        fp[cls] = fp_cls
        fn[cls] = fn_cls
        tn[cls] = tn_cls

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    """
    自定义实现的Soft Dice损失，支持多类别、批次级Dice计算
    与nnUNetv2的SoftDiceLoss接口保持一致
    """

    def __init__(
            self,
            apply_nonlin: callable = torch.softmax,  # 用于将logits转换为概率的函数
            batch_dice: bool = True,  # 是否在批次维度上计算Dice
            do_bg: bool = True,  # 是否包含背景类
            smooth: float = 1e-5  # 平滑项，避免除以零
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: 模型输出的logits，形状为 [B, C, H, W]
            target: one-hot编码的标签，形状为 [B, C, H, W]
        Returns:
            dice_loss: 计算得到的Dice损失
        """
        # 1. 将logits转换为概率（如softmax）
        if self.apply_nonlin is not None:
            input = self.apply_nonlin(input)

        # 2. 确保输入和目标形状一致
        assert input.shape == target.shape, f"输入形状 {input.shape} 与目标形状 {target.shape} 不匹配"
        assert input.dim() == 4, f"输入必须是4维张量 [B, C, H, W]，但得到 {input.dim()} 维"

        # 3. 计算交并集（flatten空间维度）
        input_flat = torch.flatten(input, start_dim=2)  # [B, C, H*W]
        target_flat = torch.flatten(target, start_dim=2)  # [B, C, H*W]

        # 4. 计算交集和并集
        intersection = (input_flat * target_flat).sum(dim=2)  # [B, C]
        union = input_flat.sum(dim=2) + target_flat.sum(dim=2)  # [B, C]

        # 5. 批次级Dice计算（在批次维度上求和）
        if self.batch_dice:
            intersection = intersection.sum(dim=0)  # [C]
            union = union.sum(dim=0)  # [C]

        # 6. 计算Dice系数（添加平滑项）
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 7. 排除背景类（如果需要）
        if not self.do_bg:
            if dice.shape[0] > 1:
                dice = dice[1:]  # 去掉背景类（索引0）
            else:
                warnings.warn("当do_bg=False时，至少需要2个类别，否则无法排除背景类")

        # 8. 计算损失（1 - 平均Dice系数）
        dice_loss = 1.0 - dice.mean()
        return dice_loss
