from PIL import Image

import json


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
