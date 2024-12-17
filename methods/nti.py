from typing import Union, Optional, Tuple, List, Dict
from tqdm import tqdm
import torch
import cv2  # OpenCV库，用于图像处理和计算机视觉任务
import numpy as np
from PIL import Image
from IPython.display import display
from torch.optim import Adam


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
):
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(
    images,  # 输入的图像数据，可以是图像列表或四维张量，包含多张图像
    num_rows=1,  # 每行显示的图像数量，默认为1
    offset_ratio=0.02,  # 图像之间的间距比例，相对于图像高度的百分比
):
    if type(images) is list:  # 如果传入的 images 是列表
        num_empty = len(images) % num_rows  # 计算每行剩余空缺图像的数量（用于填补空位）
    elif (
        images.ndim == 4
    ):  # 如果传入的 images 是四维张量 (batch_size, height, width, channels)
        num_empty = images.shape[0] % num_rows  # 同样计算每行空缺的图像数
    else:  # 如果 images 既不是列表，也不是四维张量
        images = [images]  # 将其转换为列表
        num_empty = 0  # 没有空缺

    # 创建与图像尺寸相同的空白图像，填充颜色为白色 (255)
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    # 将 images 转换为 uint8 类型，并根据 num_empty 填充空白图像
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)  # 图像的总数，包括填充的空白图像

    h, w, c = images[0].shape  # 获取单张图像的高度 (h)、宽度 (w) 和通道数 (c)
    offset = int(h * offset_ratio)  # 根据图像高度和 offset_ratio 计算图像之间的偏移量
    num_cols = num_items // num_rows  # 计算每行显示的图像列数

    # 创建一个空白大图像，用来拼接所有小图像
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),  # 总高度包括偏移
                w * num_cols + offset * (num_cols - 1),  # 总宽度包括偏移
                3,  # 通道数（彩色图像）
            ),
            dtype=np.uint8,
        )
        * 255  # 填充为白色背景
    )

    # 遍历每一行和每一列，将图像放置在正确的位置
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,  # 放置在行的正确位置
                j * (w + offset) : j * (w + offset) + w,  # 放置在列的正确位置
            ] = images[
                i * num_cols + j
            ]  # 将图像插入到大图像中

    # 将拼接后的大图像转换为 PIL 图像
    pil_img = Image.fromarray(image_)
    # 在 Jupyter Notebook 中显示图像
    display(pil_img)


def diffusion_step(
    model, controller, latents, context, t, guidance_scale, low_resource=False
):
    # 如果启用低资源模式
    if low_resource:
        # 使用无条件的上下文 (context[0]) 计算噪声预测
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]
        # 使用文本条件的上下文 (context[1]) 计算噪声预测
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )["sample"]
    else:
        # 将潜变量复制一份（无条件 + 有条件），用于同时处理无条件和有条件的扩散
        latents_input = torch.cat([latents] * 2)
        # 通过模型的 U-Net 预测噪声，使用完整的上下文（包含无条件和有条件）
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        # 将预测的噪声分割为无条件部分和有条件部分
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    # 通过引导系数 `guidance_scale` 调整噪声预测
    # 引导公式：无条件噪声 + (有条件噪声 - 无条件噪声) * guidance_scale
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )

    # 使用模型的调度器（scheduler）更新潜变量
    # 根据当前步 `t` 和调整后的噪声预测 `noise_pred`，更新潜变量
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]

    # 调用控制器的回调函数，可能用于监控或进一步修改潜变量
    latents = controller.step_callback(latents)

    # 返回更新后的潜变量
    return latents


def latent2image(vae, latents):
    # 1 / 0.18215 是一个缩放系数，用于对潜变量进行缩放
    latents = 1 / 0.18215 * latents

    # 使用 VAE（变分自编码器）的解码器将潜变量转换为图像
    image = vae.decode(latents)["sample"]

    # 对解码后的图像进行归一化处理，调整像素值到 [0, 1] 范围内
    image = (image / 2 + 0.5).clamp(0, 1)

    # 将图像从 GPU（如果使用）转移到 CPU，并调整维度顺序
    # 原来的维度是 (batch_size, channels, height, width)，通过 permute 调整为 (batch_size, height, width, channels)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    # 将图像从 [0, 1] 归一化的浮点数转换为 [0, 255] 范围的整数像素值，并将类型转换为 uint8
    image = (image * 255).astype(np.uint8)

    # 返回转换后的图像
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    # 如果初始的潜变量 latent 为空，生成一个随机的潜变量
    if latent is None:
        latent = torch.randn(  # 生成随机的正态分布张量作为潜变量
            (
                1,
                model.unet.config.in_channels,
                height // 8,
                width // 8,
            ),  # 形状为(1, 通道数, 高度/8, 宽度/8)
            generator=generator,  # 使用给定的随机生成器控制随机性
        )

    # 将潜变量扩展为批量大小，用于同时处理多张图像
    latents = latent.expand(
        batch_size,  # 将潜变量扩展到 batch_size 大小
        model.unet.config.in_channels,  # 使用 UNet 模型的输入通道数
        height // 8,  # 图像高度缩小 8 倍
        width // 8,  # 图像宽度缩小 8 倍
    ).to(
        model.device
    )  # 将张量移动到模型的设备上（如GPU）

    # 返回初始潜变量和扩展后的潜变量
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.0,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt, padding="max_length", max_length=77, return_tensors="pt"
    )
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vqvae, latents)

    return image, latent


@torch.no_grad()  # 禁用梯度计算，提高推理速度并减少内存占用
def text2image_ldm_stable(
    model,  # 深度学习模型，用于生成图像
    prompt: List[str],  # 提供的文本提示，用于生成相应图像
    controller,  # 控制器，用于控制注意力机制或其他自定义操作
    num_inference_steps: int = 50,  # 推理步骤的数量，默认为50，越多质量越高
    guidance_scale: float = 7.5,  # 引导尺度，控制文本生成图像时的条件强度
    generator: Optional[torch.Generator] = None,  # 随机数生成器，用于控制随机性
    latent: Optional[torch.FloatTensor] = None,  # 初始潜变量，若无则生成新的
    low_resource: bool = False,  # 低资源模式，是否启用更节省资源的推理过程
):
    register_attention_control(model, controller)  # 注册控制器到模型的注意力机制中
    height = width = 512  # 图像的高度和宽度设置为512x512
    batch_size = len(prompt)  # 批量大小等于提示文本的数量

    # 对文本提示进行编码，并返回用于生成图像的嵌入向量
    text_input = model.tokenizer(
        prompt,  # 提供的文本提示
        padding="max_length",  # 填充文本至最大长度
        max_length=model.tokenizer.model_max_length,  # 模型支持的最大长度
        truncation=True,  # 如果超过最大长度则截断
        return_tensors="pt",  # 返回PyTorch张量格式
    )
    # 获取文本嵌入，将其传入模型的文本编码器中
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]  # 获取文本序列的最大长度

    # 为无条件输入创建占位符，并将其编码为嵌入
    uncond_input = model.tokenizer(
        [""] * batch_size,  # 生成与批量大小相同的空白文本输入
        padding="max_length",  # 填充至最大长度
        max_length=max_length,  # 使用与文本提示相同的最大长度
        return_tensors="pt",  # 返回PyTorch张量格式
    )
    # 获取无条件嵌入
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    # 创建上下文，包括无条件和有条件的文本嵌入
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:  # 如果不处于低资源模式
        context = torch.cat(context)  # 将嵌入拼接在一起
    # 在低资源模式下，代码并没有将 context 拼接在一起，而是保持 context 作为一个包含两个元素的列表（无条件嵌入和有条件嵌入）。这意味着在后续的 diffusion_step 函数中，模型会分别使用这两个嵌入来处理无条件和有条件的生成，而不需要在内存中加载一个更大的拼接后的张量。这种方式减少了内存占用，因此在资源较少的情况下，更加高效。
    # 初始化潜变量
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # 设置推理的时间步数
    model.scheduler.set_timesteps(num_inference_steps)
    # 对于每一个推理步骤，更新潜变量
    for t in tqdm(model.scheduler.timesteps):  # 通过进度条显示推理进度
        latents = diffusion_step(
            model, controller, latents, context, t, guidance_scale, low_resource
        )

    # 使用潜变量生成最终图像
    image = latent2image(model.vae, latents)

    # 返回生成的图像和潜变量
    return image, latent


def register_attention_control(
    pipe, controller
):  # 注册注意力控制机制，允许控制注意力分布
    def ca_forward(self, place_in_unet):  # 定义一个前向传播函数，用于自定义的注意力机制
        to_out = self.to_out  # 获取输出层
        if (
            type(to_out) is torch.nn.modules.container.ModuleList
        ):  # 如果输出层是ModuleList类型
            to_out = self.to_out[0]  # 取第一个输出层
        else:
            to_out = self.to_out  # 否则直接使用输出层

        def forward(  # 定义前向传播函数
            hidden_states,  # 输入的隐藏状态
            encoder_hidden_states=None,  # 编码器隐藏状态（用于跨注意力机制）
            attention_mask=None,  # 注意力掩码
            temb=None,  # 时间嵌入
        ):
            is_cross = encoder_hidden_states is not None  # 判断是否为跨注意力机制

            residual = hidden_states  # 保存原始隐藏状态用于残差连接

            if self.spatial_norm is not None:  # 如果存在空间归一化
                hidden_states = self.spatial_norm(
                    hidden_states, temb
                )  # 对隐藏状态进行归一化

            input_ndim = hidden_states.ndim  # 获取输入的维度

            if input_ndim == 4:  # 如果输入是四维张量（常见于图像处理）
                batch_size, channel, height, width = hidden_states.shape  # 获取形状信息
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(
                    1, 2
                )  # 变换形状用于计算注意力

            batch_size, sequence_length, _ = (
                hidden_states.shape  # 获取隐藏状态的形状
                if encoder_hidden_states
                is None  # 如果没有编码器隐藏状态，使用当前隐藏状态的形状
                else encoder_hidden_states.shape  # 否则使用编码器隐藏状态的形状
            )
            attention_mask = self.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )  # 准备注意力掩码

            if self.group_norm is not None:  # 如果存在组归一化
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(
                    1, 2
                )  # 对隐藏状态进行归一化

            query = self.to_q(hidden_states)  # 计算查询向量

            if encoder_hidden_states is None:  # 如果没有编码器隐藏状态
                encoder_hidden_states = hidden_states  # 使用隐藏状态作为编码器隐藏状态
            elif self.norm_cross:  # 如果需要归一化跨注意力
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )  # 归一化编码器隐藏状态

            key = self.to_k(encoder_hidden_states)  # 计算键向量
            value = self.to_v(encoder_hidden_states)  # 计算值向量

            query = self.head_to_batch_dim(query)  # 将查询向量转换为多头形式
            key = self.head_to_batch_dim(key)  # 将键向量转换为多头形式
            value = self.head_to_batch_dim(value)  # 将值向量转换为多头形式

            attention_probs = self.get_attention_scores(
                query, key, attention_mask
            )  # 计算注意力分数
            attention_probs = controller(
                attention_probs, is_cross, place_in_unet
            )  # 调用控制器调整注意力分数

            hidden_states = torch.bmm(
                attention_probs, value
            )  # 根据注意力分数加权计算新的隐藏状态
            hidden_states = self.batch_to_head_dim(
                hidden_states
            )  # 将多头注意力结果转换回原始维度

            hidden_states = to_out(hidden_states)  # 通过线性层输出结果

            if input_ndim == 4:  # 如果输入是四维张量
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )  # 恢复形状

            if self.residual_connection:  # 如果使用残差连接
                hidden_states = hidden_states + residual  # 加上原始隐藏状态

            hidden_states = hidden_states / self.rescale_output_factor  # 对输出进行缩放

            return hidden_states  # 返回新的隐藏状态

        return forward  # 返回前向传播函数

    class DummyController:  # 定义一个默认的控制器类

        def __call__(self, *args):  # 调用控制器时，直接返回输入的注意力分数
            return args[0]

        def __init__(self):  # 初始化控制器
            self.num_att_layers = 0  # 记录注意力层的数量

    if controller is None:  # 如果没有提供控制器
        controller = DummyController()  # 使用默认的控制器

    def register_recr(net_, count, place_in_unet):  # 递归地注册注意力控制
        if net_.__class__.__name__ == "Attention":  # 如果当前子网络是注意力层
            net_.forward = ca_forward(net_, place_in_unet)  # 替换其前向传播函数
            return count + 1  # 增加注意力层计数
        elif hasattr(net_, "children"):  # 如果子网络包含子模块
            for net__ in net_.children():  # 递归处理每个子模块
                count = register_recr(net__, count, place_in_unet)
        return count  # 返回注意力层的计数

    cross_att_count = 0  # 初始化跨注意力层的计数
    sub_nets = pipe.unet.named_children()  # 获取UNet模型的所有子网络
    for net in sub_nets:  # 遍历每个子网络
        if "down" in net[0]:  # 如果子网络属于下采样部分
            cross_att_count += register_recr(
                net[1], 0, "down"
            )  # 注册下采样部分的注意力控制
        elif "up" in net[0]:  # 如果子网络属于上采样部分
            cross_att_count += register_recr(
                net[1], 0, "up"
            )  # 注册上采样部分的注意力控制
        elif "mid" in net[0]:  # 如果子网络属于中间部分
            cross_att_count += register_recr(
                net[1], 0, "mid"
            )  # 注册中间部分的注意力控制

    controller.num_att_layers = cross_att_count  # 设置控制器中的注意力层数量


def get_word_inds(text: str, word_place: int | str, tokenizer):
    # 将传入的文本按空格分割为单词列表
    split_text = text.split(" ")

    # 如果 word_place 是字符串，找到字符串在 split_text 中的位置
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    # 如果 word_place 是整数，直接将其转为列表，表示特定单词的位置
    elif type(word_place) is int:
        word_place = [word_place]

    # 初始化一个空的列表，用来存储结果
    out = []

    # 如果指定的单词位置不为空，进行进一步处理
    if len(word_place) > 0:
        # 对文本进行编码并解码，去除 "#"
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][
            1:-1
        ]  # 去掉首尾特殊符号

        # 初始化两个变量，cur_len 用来记录当前单词长度，ptr 用来指向目标单词的位置
        cur_len, ptr = 0, 0

        # 遍历编码后的单词列表
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])  # 累加当前单词的长度
            if ptr in word_place:  # 如果当前指针位置是目标单词位置
                out.append(i + 1)  # 将该单词的索引添加到输出列表中
            if cur_len >= len(split_text[ptr]):  # 如果当前长度超过了原始单词长度
                ptr += 1  # 移动指针到下一个单词
                cur_len = 0  # 重置当前长度

    # 返回结果列表，转换为 numpy 数组
    return np.array(out)


def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=77,
):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i
        )
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [
                get_word_inds(prompts[i], key, tokenizer)
                for i in range(1, len(prompts))
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(
                        alpha_time_words, item, i, ind
                    )
    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )
    return alpha_time_words


import torch
import torch.nn.functional as nnf
import numpy as np
from PIL import Image
import abc
from typing import Union, Optional, List, Tuple, Dict
from tqdm import tqdm

NUM_DDIM_STEPS = 50
MAX_NUM_WORDS = 77
LOW_RESOURCE = False
GUIDANCE_SCALE = 7.5


@torch.no_grad()
def text2image_ldm_stable(
    pipe,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type="image",
):
    batch_size = len(prompt)
    register_attention_control(pipe, controller)
    height = width = 512

    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = pipe.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings_ = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[
            0
        ]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, pipe, height, width, generator, batch_size)
    pipe.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(pipe.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat(
                [uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings]
            )
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = diffusion_step(
            pipe, controller, latents, context, t, guidance_scale, low_resource=False
        )

    if return_type == "image":
        image = latent2image(pipe.vae, latents)
    else:
        image = latents
    return image, latent


def null_text_generation(
    pipe,
    prompts,
    controller,
    latent=None,
    generator=None,
    uncond_embeddings=None,
    verbose=True,
):
    images, x_t = text2image_ldm_stable(
        pipe,
        prompts,
        controller,
        latent=latent,
        num_inference_steps=NUM_DDIM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        uncond_embeddings=uncond_embeddings,
    )
    if verbose:
        view_images(images)
    return images, x_t


def run_and_display(
    pipe,
    prompts,
    controller,
    latent=None,
    generator=None,
    uncond_embeddings=None,
    verbose=True,
):
    print("ddim generation")
    baseline_images, _ = text2image_ldm_stable(
        pipe,
        prompts,
        EmptyControl(),
        latent=latent,
        generator=generator,
    )
    print("null text generation")
    images, x_t = text2image_ldm_stable(
        pipe,
        prompts,
        controller,
        latent=latent,
        num_inference_steps=NUM_DDIM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        uncond_embeddings=uncond_embeddings,
    )
    if verbose:
        view_images(images)
    return baseline_images, images, x_t


class LocalBlend:

    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [
                item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS)
                for item in maps
            ]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(x_t, maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(x_t, maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(
        self,
        prompts: List[str],
        words: List[List[str]],
        substruct_words=None,
        start_blend=0.2,
        th=(0.3, 0.3),
        pipe=None,
    ):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, pipe.tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, pipe.tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(pipe.device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(pipe.device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th = th


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, pipe):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.tokenizer = pipe.tokenizer
        self.device = pipe.device


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, pipe):
        super().__init__(pipe)
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32**2:
            attn_base = attn_base.unsqueeze(0).expand(
                att_replace.shape[0], *attn_base.shape
            )
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (
            self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        ):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(
                    attn_base, attn_repalce, place_in_unet
                )
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Tuple[float, float]]
        ],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        pipe=None,
    ):
        super(AttentionControlEdit, self).__init__(pipe)
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, self.tokenizer
        ).to(self.device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(
            num_steps * self_replace_steps[1]
        )
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        pipe=None,
    ):
        super().__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            pipe,
        )
        self.mapper = get_replacement_mapper(prompts, self.tokenizer).to(self.device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        pipe=None,
    ):
        super().__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            pipe,
        )
        self.mapper, alphas = get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(
                attn_base, att_replace
            )
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
        pipe=None,
    ):
        super(AttentionReweight, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            pipe,
        )
        self.equalizer = equalizer.to(self.device)
        self.prev_controller = controller


def get_equalizer(
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    values: Union[List[float], Tuple[float, ...]],
    tokenizer=None,
):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def aggregate_attention(
    prompts,
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    is_cross: bool,
    select: int,
):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[
                    select
                ]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(
    prompts: List[str],
    is_replace_controller: bool,
    cross_replace_steps: Dict[str, float],
    self_replace_steps: float,
    blend_words=None,
    equilizer_params=None,
    pipe=None,
) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, pipe=pipe)
    if is_replace_controller:
        controller = AttentionReplace(
            prompts,
            NUM_DDIM_STEPS,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=lb,
            pipe=pipe,
        )
    else:
        controller = AttentionRefine(
            prompts,
            NUM_DDIM_STEPS,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=lb,
            pipe=pipe,
        )
    if equilizer_params is not None:
        eq = get_equalizer(
            prompts[1],
            equilizer_params["words"],
            equilizer_params["values"],
            pipe.tokenizer,
        )
        controller = AttentionReweight(
            prompts,
            NUM_DDIM_STEPS,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            equalizer=eq,
            local_blend=lb,
            controller=controller,
            pipe=pipe,
        )
    return controller


def show_cross_attention(
    prompts: List[str],
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    tokenizer,
    select: int = 0,
):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(
        prompts, attention_store, res, from_where, True, select
    )
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    view_images(np.stack(images, axis=0))


def show_self_attention_comp(
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    max_com=10,
    select: int = 0,
):
    attention_maps = (
        aggregate_attention(attention_store, res, from_where, False, select)
        .numpy()
        .reshape((res**2, res**2))
    )
    u, s, vh = np.linalg.svd(
        attention_maps - np.mean(attention_maps, axis=1, keepdims=True)
    )
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1))


# 这个函数用于加载图像，并根据给定的左、右、上、下边界进行裁剪，最后将图像调整为512x512像素。
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    # 正常来说，就是取 image[top:h-bottom, left:w-right]
    # left = min(left, w - 1)
    # right = min(right, w - left - 1)
    # top = min(top, h - 1)
    # bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    # 中心裁剪,使其变成正方形
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:

    # 在逆向步骤中计算给定时间步长的上一个潜在变量（latent）。用于图像生成过程中逐步回溯到原始图像的潜在表示。
    def prev_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )
        return prev_sample

    # 计算给定时间步长的下一个潜在变量。用于图像生成过程中逐步前向推导到生成最终图像。
    def next_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    # 从模型的UNet部分获取单次噪声预测，用于生成过程中的去噪操作。
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    # 获取噪声预测，并根据前向或逆向操作来更新潜在变量。
    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.pipe.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    # 将潜在变量转换为图像。
    # 通过vae之后的latent的分布有差异，为了使其更接近标准高斯分布，遂采用了这样一个因子来rescale
    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.pipe.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    # 将图像转换为潜在变量。
    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.pipe.vae.encode(image)["latent_dist"].mean
                latents = latents * 0.18215
        return latents

    # 初始化文本提示（prompt），为生成任务做好准备。
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipe.text_encoder(
            uncond_input.input_ids.to(self.pipe.device)
        )[0]
        text_input = self.pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(
            text_input.input_ids.to(self.pipe.device)
        )[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    # 通过DDIM去噪过程，逐步将潜在变量转换为图像的潜在表示。
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.pipe.scheduler.timesteps[
                len(self.pipe.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.pipe.scheduler

    # 通过DDIM反向推导，将图像转换为潜在变量，并进行图像重建。
    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            t = self.pipe.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(
                    latent_cur, t, cond_embeddings
                )
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(
                    latent_cur, t, uncond_embeddings
                )
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    def invert(
        self,
        image: Image,
        prompt: str,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
        verbose=False,
    ):
        self.init_prompt(prompt)
        register_attention_control(self.pipe, None)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(
            ddim_latents, num_inner_steps, early_stop_epsilon
        )
        return (image, image_rec), ddim_latents[-1], uncond_embeddings

    def __init__(self, pipe):
        self.pipe = pipe
        self.device = pipe.device
        self.tokenizer = self.pipe.tokenizer
        self.pipe.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


# 总的来说，这段代码实现了一个序列比对的流程，基于动态规划的思想，能够生成比对矩阵、回溯矩阵，并根据比对结果生成映射关系，用于自然语言处理任务中的序列比对和替换映射。


import torch
import numpy as np


# 用于存储序列比对中的 gap（插入空位）、match（匹配得分）、mismatch（不匹配惩罚）等参数。
class ScoreParams:

    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


# 生成一个大小为 (size_x + 1, size_y + 1) 的二维矩阵，并初始化第 0 行和第 0 列为 gap 值。用于动态规划算法的矩阵初始化。
def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


# 生成一个用于回溯比对路径的矩阵。第一行和第一列分别初始化为 1 和 2，矩阵左上角初始化为 4。这个矩阵记录了路径追踪信息，帮助最终回溯序列。
def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


# 执行全局比对算法。通过比较字符串 x 和 y，根据 ScoreParams 中的 gap、match、mismatch 参数填充得分矩阵和回溯矩阵。结果返回得分矩阵和回溯矩阵。
def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


# 根据回溯矩阵重新构造比对后的序列。返回两个比对后的序列 x_seq 和 y_seq 以及序列位置映射 mapper_y_to_x
def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append("-")
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i - 1])
            y_seq.append("-")
            i = i - 1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


# 对字符串 x 和 y 进行比对，并生成一个映射矩阵和 alphas（权重数组），用于标识比对的结果。比对使用的字符是通过 tokenizer 编码的。
def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0] :] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


# 对于一组 prompts，计算它们之间的映射矩阵。用于将一系列序列进行比对并生成相应的映射。
def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)


# 找到指定单词或位置在文本中的索引。将文本编码后，返回目标单词在编码序列中的索引位置。
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


# 生成两个相同长度的文本 x 和 y 之间的替换映射矩阵，表示在 x 和 y 中不同单词的对应关系。
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(" ")
    words_y = y.split(" ")
    if len(words_x) != len(words_y):
        raise ValueError(
            f"attention replacement edit can only be applied on prompts with the same length"
            f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
        )
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()


# 对一组 prompts 生成替换映射矩阵，用于将文本中的不同位置替换映射到目标文本的对应位置。
def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)
