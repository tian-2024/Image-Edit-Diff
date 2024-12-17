from PIL import Image
import matplotlib.pyplot as plt
import json

import torch
import torchvision.transforms as tfms


# 读取 imgs.json 文件并提取图像路径和描述
def load_image_data(json_file: str):
    # 打开并读取 JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取图像路径和描述
    images = []
    for item in data.get("images", []):
        path = item.get("path")
        source_caption = item.get("source_caption")
        target_caption = item.get("target_caption")
        images.append(
            {
                "path": path,
                "source_caption": source_caption,
                "target_caption": target_caption,
            }
        )

    return images


def load_image(image_path: str):
    image = Image.open(image_path)
    return image


def load_pipe(device):
    import os

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
    )
    pipe.to(device)
    return pipe


def display_images(image_list, figsize=(15, 5)):
    """
    显示图像列表，每个图像显示在一排中。

    :param image_list: PIL Image 对象的列表
    :param figsize: 图像的整体显示大小
    """
    n = len(image_list)

    # 创建一个包含 n 个子图的图形
    fig, axes = plt.subplots(1, n, figsize=figsize)

    # 如果只有一个图像，`axes` 会是一个单独的轴，而不是一个数组
    if n == 1:
        axes = [axes]

    for i, img in enumerate(image_list):
        if isinstance(img, Image.Image):  # 确保每个元素是 PIL Image 对象
            axes[i].imshow(img)
            axes[i].axis("off")  # 关闭坐标轴
        else:
            raise ValueError("Each item in the list should be a PIL Image object.")

    plt.tight_layout()  # 自动调整子图间的间距
    plt.show()


def pil_to_tensor(
    image: Image.Image, device: torch.device, dtype=torch.float32
) -> torch.Tensor:
    """
    将PIL图像转换为预处理后的张量。

    参数:
        image (PIL.Image.Image): 输入的PIL图像。
        device (torch.device): 要将张量移动到的设备（例如，'cuda' 或 'cpu'）。

    返回:
        torch.Tensor: 预处理后的图像张量，形状为 [1, C, H, W]，值范围为 [-1, 1]。
    """
    # 定义转换：将图像转换为张量
    to_tensor = tfms.ToTensor()

    # 应用转换并添加批次维度
    tensor = to_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)

    # 将像素值从 [0, 1] 范围缩放到 [-1, 1]
    tensor = tensor * 2 - 1

    return tensor
