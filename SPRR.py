# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

# 可添加使用 OIR
from OIR import clean_ddr_illumination

def retinex_decompose(img, sigma_list=[15, 80, 250], kernel_size=61):
    """
    PyTorch版本的多尺度Retinex分解 在GPU上运行
    
    Args:
        img: Tensor, [C, H, W], 0 ~ 1
        sigma_list: 高斯模糊的标准差列表
        kernel_size: 高斯核大小 0 表示自动计算
    
    Returns:
        reflectance, illumination (Tensor)
    """
    # 确保输入在有效范围内
    img = torch.clamp(img, 1e-6, 1.0)
    
    # 转换到log空间 [C, H, W]
    log_img = torch.log(img)
    
    # 初始化反射分量
    retinex = torch.zeros_like(log_img)
    
    # 对每个尺度进行处理
    for sigma in sigma_list:
        # kernel_size = int(2 * torch.ceil(torch.tensor(2.0 * sigma)) + 1)
        # 应用高斯模糊 [C, H, W]
        blur = gaussian_blur2d(img, kernel_size, sigma)
        blur = torch.clamp(blur, 1e-6, 1.0)  # 避免log(0)
        # 累积反射分量
        retinex += log_img - torch.log(blur)
    
    # 平均多尺度结果
    retinex /= len(sigma_list)
    
    # 计算光照分量（log域）
    illumination_log = log_img - retinex
    
    # 转换回线性空间
    reflectance = torch.exp(retinex)
    illumination = torch.exp(illumination_log)
    
    # 确保输出在有效范围内
    reflectance = torch.clamp(reflectance, 0, 1)
    illumination = torch.clamp(illumination, 0, 1)
    
    return reflectance, illumination

def gaussian_blur2d(img, kernel_size, sigma):
    """
    对2D图像应用高斯模糊（支持批量处理）
    """
    # 创建高斯核
    kernel = get_gaussian_kernel2d(kernel_size, sigma, img.device, img.dtype)
    
    # 对每个通道分别应用卷积
    channels = []
    for c in range(img.shape[0]):
        channel = img[c:c+1]  # [1, H, W]
        blurred = F.conv2d(
            channel.unsqueeze(0),  # [1, 1, H, W]
            kernel.unsqueeze(0).unsqueeze(0),  # [1, 1, kernel_size, kernel_size]
            padding=kernel_size//2
        )
        channels.append(blurred.squeeze(0))
    
    return torch.cat(channels, dim=0)

def get_gaussian_kernel2d(kernel_size, sigma, device, dtype):
    """
    生成2D高斯核
    """
    # 创建坐标网格
    x = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, device=device, dtype=dtype)
    y = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, device=device, dtype=dtype)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    
    # 计算高斯分布
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    return kernel

def swap_retinex_illumination2(img_a, img_b):
    refl_a, _ = retinex_decompose(img_a)
    _, illum_b = retinex_decompose(img_b)

    illum_b = clean_ddr_illumination(
        ddr_tensor=img_b.unsqueeze(0),
        illumination_b=illum_b.unsqueeze(0),
    ).squeeze(0)

    return torch.clamp(refl_a*illum_b, 0, 1)

def random_domain_retinex_swap_extra(train_img, extra_img):
    extra_img = extra_img.to(device=train_img.device, dtype=train_img.dtype)
    H, W = train_img.shape[2], train_img.shape[3]   # 取出目标尺寸
    extra_img_upsampled = F.interpolate(
        extra_img,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )

    for b in range(train_img.size(0)):
        train_img[b] = swap_retinex_illumination2(
            train_img[b],
            extra_img_upsampled[b],
        )

    return train_img