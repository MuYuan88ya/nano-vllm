import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认的权重加载函数

    直接将加载的权重复制到参数中（简单的全量复制）

    参数：
        param: 目标参数（模型中的参数）
        loaded_weight: 加载的权重张量
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从 HuggingFace 格式的 checkpoint 加载模型权重

    支持特性：
    1. SafeTensors 格式（安全的张量序列化格式）
    2. 权重名称映射（处理合并的参数，如 QKV projection）
    3. 自定义 weight_loader（处理张量并行等特殊加载逻辑）

    参数：
        model: 要加载权重的模型
        path: checkpoint 目录路径（包含 .safetensors 文件）

    权重映射机制：
    某些模型将多个权重合并为一个参数（如 QKV projection），
    需要通过 packed_modules_mapping 将原始权重名映射到合并后的参数名

    示例：
        原始权重: q_proj, k_proj, v_proj
        合并参数: qkv_proj
        映射: {"q_proj": ("qkv_proj", "q"), ...}

    自定义 weight_loader：
    参数可以定义自己的 weight_loader 方法，用于处理特殊的加载逻辑：
    - 张量并行：只加载部分权重
    - 权重重排：调整权重布局
    - 数据类型转换：转换为目标数据类型
    """
    # 获取模型的权重映射表（如果有）
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 遍历所有 safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重
            for weight_name in f.keys():
                # 检查是否需要权重名称映射
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 需要映射：获取目标参数名和分片 ID
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 使用自定义的 weight_loader（处理合并权重）
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 不需要映射：直接加载
                    param = model.get_parameter(weight_name)
                    # 获取 weight_loader（可能是自定义的，也可能是默认的）
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
