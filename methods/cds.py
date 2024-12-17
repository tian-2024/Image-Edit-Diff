import torch
from torch import optim
from tqdm import tqdm
from typing import Optional, Union, List, Callable
from PIL import Image
import numpy as np
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention


class MyCrossAttnProcessor:
    def __call__(
        self,
        attn: Attention,  # Attention 模块的实例
        hidden_states,  # 当前隐藏状态，形状为 [batch_size, sequence_length, hidden_dim]
        encoder_hidden_states=None,  # 编码器的隐藏状态（用于跨注意力），默认为 None
        attention_mask=None,  # 注意力掩码，默认为 None
    ):
        # 获取批次大小和序列长度
        batch_size, sequence_length, _ = hidden_states.shape

        # 将隐藏状态通过 Attention 模块的查询（query）线性层
        query = attn.to_q(hidden_states)

        # 如果提供了编码器隐藏状态，则使用它作为键（key）和值（value）的输入，否则使用当前隐藏状态
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)  # 通过线性层生成键
        value = attn.to_v(encoder_hidden_states)  # 通过线性层生成值

        # 将查询、键和值的形状调整为适应多头注意力机制
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算注意力得分，并应用 softmax 获得注意力概率分布
        attention_probs = attn.get_attention_scores(query, key)

        # 将注意力概率分布与值进行批量矩阵乘法，得到新的隐藏状态
        hidden_states = torch.bmm(attention_probs, value)
        # 将多头注意力的维度还原回原始形状
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 通过输出线性层进行投影
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout 层进行正则化
        hidden_states = attn.to_out[1](hidden_states)

        # 保存文本条件的注意力图
        # 如果隐藏状态的第一个维度为4，表示处理参考（ref）图像的注意力
        if hidden_states.shape[0] == 4:
            attn.hs = hidden_states[2:3]
        # 否则，表示处理目标（trg）图像的注意力
        else:
            attn.hs = hidden_states[1:2]

        # 返回更新后的隐藏状态
        return hidden_states


class CutLoss:
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = n_patches  # 设置每张图像要采样的补丁数量
        self.patch_size = patch_size  # 设置补丁的大小（可以是一个整数或列表）

    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0  # 初始化总损失为0

        bs, res2, c = (
            ref_noise.shape
        )  # 获取参考噪声的批次大小（bs）、分辨率平方（res2）和通道数（c）
        res = int(np.sqrt(res2))  # 计算图像的分辨率（假设图像是正方形）

        # 将参考噪声和目标噪声从 [batch, res2, c] 形状重塑为 [batch, c, res, res]
        ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)
        trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)

        # 遍历所有指定的补丁大小
        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(
                    kernel_size=(ps, ps)
                )  # 定义一个平均池化层，窗口大小为 (ps, ps)
                ref_noise_pooled = pooling(ref_noise_reshape)  # 对参考噪声进行池化
                trg_noise_pooled = pooling(trg_noise_reshape)  # 对目标噪声进行池化
            else:
                ref_noise_pooled = ref_noise_reshape  # 如果补丁大小为1，不进行池化
                trg_noise_pooled = trg_noise_reshape

            # 对池化后的噪声进行归一化，使每个通道的向量长度为1
            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            # 调整维度顺序并展平空间维度，以便后续处理
            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(
                1, 2
            )  # [batch, res*res, c]
            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(
                1, 2
            )  # [batch, res*res, c]

            # 随机打乱补丁索引
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1])
            # 选择前 n_patches 个补丁，确保不超过实际补丁数量
            patch_ids = patch_ids[: int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            # 将补丁索引转换为张量，并移动到与参考噪声相同的设备上
            patch_ids = torch.tensor(
                patch_ids, dtype=torch.long, device=ref_noise.device
            )

            # 从参考噪声中选择指定的补丁，并展平为 [n_patches, c]
            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            # 从目标噪声中选择相同索引的补丁，并展平为 [n_patches, c]
            trg_sample = trg_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            # 计算 PatchNCE 损失并累加到总损失中
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean()

        return loss  # 返回总损失

    # PatchNCE loss from https://github.com/taesungp/contrastive-unpaired-translation
    # https://github.com/YSerin/ZeCon/blob/main/optimization/losses.py
    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T=0.07):
        # 设置批次大小和温度参数
        batch_size = batch_size
        nce_T = nce_T

        # 定义交叉熵损失函数，不进行归约，以便后续处理
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        mask_dtype = torch.bool  # 定义掩码的数据类型为布尔型

        num_patches = ref_noise.shape[0]  # 获取补丁数量
        dim = ref_noise.shape[1]  # 获取特征维度
        ref_noise = ref_noise.detach()  # 分离参考噪声，防止梯度传播

        # 计算正样本的相似度（点积）
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)  # 重塑为 [num_patches, 1]

        # 将特征重新形状为 [batch_size, npatches, dim]
        ref_noise = ref_noise.view(batch_size, -1, dim)  # 参考噪声
        trg_noise = trg_noise.view(batch_size, -1, dim)  # 目标噪声
        npatches = ref_noise.shape[1]  # 每批次的补丁数

        # 计算负样本的相似度（参考噪声与目标噪声的点积）
        l_neg_curbatch = torch.bmm(
            ref_noise, trg_noise.transpose(2, 1)
        )  # [batch_size, npatches, npatches]

        # 对角线上的相似度是同一补丁之间的相似度，没意义，因此将其填充为一个非常小的数（如 exp(-10)）
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[
            None, :, :
        ]  # 创建对角掩码
        l_neg_curbatch.masked_fill_(diagonal, -10.0)  # 将对角线元素填充为 -10.0
        l_neg = l_neg_curbatch.view(
            -1, npatches
        )  # 重塑为 [batch_size * npatches, npatches]

        # 将正样本和负样本的相似度拼接起来，并除以温度参数
        out = (
            torch.cat((l_pos, l_neg), dim=1) / nce_T
        )  # [batch_size * npatches, 1 + npatches]

        # 定义目标标签，正样本对应的标签为0
        target = torch.zeros(
            out.size(0), dtype=torch.long, device=ref_noise.device
        )  # [batch_size * npatches]

        # 计算交叉熵损失
        loss = cross_entropy_loss(out, target)  # [batch_size * npatches]

        return loss  # 返回损失


class DDSLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,
                size=(b,),
                device=z.device,
                dtype=torch.long,
            )

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep

    def get_epsilon_prediction(
        self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None
    ):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:])

        e_t = self.unet(
            latent_input,
            timestep,
            embedd,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample
        e_t_uncond, e_t = e_t.chunk(2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()

        return e_t

    def __init__(self, t_min, t_max, unet, scheduler, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scheduler = scheduler
        self.device = device


def prep_unet(unet):
    """
    准备 UNet 模型，通过冻结非注意力层的参数，并替换注意力模块的处理器为自定义的 MyCrossAttnProcessor。

    参数:
        unet (nn.Module): 要准备的 UNet 模型

    返回:
        nn.Module: 修改后的 UNet 模型
    """
    # 遍历 UNet 模型的所有参数
    for name, params in unet.named_parameters():
        if "attn1" in name:  # 如果参数名中包含 "attn1"，即自注意力层的参数
            params.requires_grad = True  # 允许这些参数参与训练
        else:
            params.requires_grad = False  # 冻结其他参数，不参与训练

    # 遍历 UNet 模型的所有子模块
    for name, module in unet.named_modules():
        module_name = type(module).__name__  # 获取子模块的类名
        if module_name == "Attention":  # 如果子模块是 Attention 模块
            module.set_processor(MyCrossAttnProcessor())  # 设置自定义的注意力处理器
    return unet  # 返回修改后的 UNet 模型


class CDSPipeline(StableDiffusionPipeline):

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        img=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            vae_magic = 0.18215
            with torch.no_grad():
                latents = self.vae.encode(img)
            latents = latents["latent_dist"].mean * vae_magic
        else:
            latents = latents.to(device)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        img: torch.Tensor = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 7.5,
        # Target prompt for editing
        trg_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        trg_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        # Additional args for CDS
        n_patches: int = 256,
        patch_size: Union[int, List[int]] = [1, 2],
        w_dds: float = 1.0,
        w_cut: float = 3.0,
    ):

        # Modify unet to save self-attention map
        self.unet = prep_unet(self.unet)

        sa_attn = {}

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input & target prompt
        prompt_embeds, trg_prompt_embeds = self._encode_prompt(
            prompt,
            trg_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            trg_prompt_embeds=trg_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            img,
        )

        # Update latents
        # timestep ~ U(0.05, 0.95) to avoid very high/low noise level
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.05)  # 50
        self.max_step = int(self.num_train_timesteps * 0.95)  # 950

        # Define loss class
        dds_loss = DDSLoss(
            t_min=self.min_step,
            t_max=self.max_step,
            unet=self.unet,
            scheduler=self.scheduler,
            device=device,
        )
        cut_loss = CutLoss(n_patches, patch_size)

        # Edit image!
        z_src = latents
        z_trg = latents.clone()
        z_trg.requires_grad = True

        optimizer = optim.SGD([z_trg], lr=0.1)

        num_warmup_steps = (
            num_inference_steps - num_inference_steps * self.scheduler.order
        )

        results = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                optimizer.zero_grad()

                z_t_src, eps, timestep = dds_loss.noise_input(
                    z_src, eps=None, timestep=None
                )
                z_t_trg, _, _ = dds_loss.noise_input(z_trg, eps, timestep)

                # get score for dds & reference attention maps
                eps_pred = dds_loss.get_epsilon_prediction(
                    torch.cat((z_t_src, z_t_trg)),
                    torch.cat((timestep, timestep)),
                    torch.cat((prompt_embeds, trg_prompt_embeds)),
                )

                eps_pred_src, eps_pred_trg = eps_pred.chunk(2)
                grad = eps_pred_trg - eps_pred_src

                sa_attn[timestep.item()] = {}

                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__

                    if module_name == "Attention":
                        if "attn1" in name and "up" in name:
                            hidden_state = module.hs
                            sa_attn[timestep.item()][name] = hidden_state.detach().cpu()

                with torch.enable_grad():
                    loss = z_trg * grad.clone()
                    # reduction 'mean'
                    loss = loss.sum() / (z_trg.shape[2] * z_trg.shape[3])

                    (2000 * loss * w_dds).backward()

                # calculate cut loss
                with torch.enable_grad():
                    z_t_trg, _, _ = dds_loss.noise_input(z_trg, eps, timestep)
                    eps_pred_trg = dds_loss.get_epsilon_prediction(
                        z_t_trg,
                        timestep,
                        trg_prompt_embeds,
                    )

                    cutloss = 0
                    for name, module in self.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "Attention":
                            # sa_cut
                            if "attn1" in name and "up" in name:
                                curr = module.hs
                                ref = sa_attn[timestep.item()][name].detach().to(device)
                                cutloss += cut_loss.get_attn_cut_loss(ref, curr)

                    (cutloss * w_cut).backward()

                optimizer.step()

                # call the callback, if provided
                if i == num_inference_steps - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if i % 50 == 0:
                    # trg img
                    img = self.decode_latents(z_trg).squeeze()
                    img = Image.fromarray((img * 255).astype(np.uint8))
                    results.append(img)

        result = self.decode_latents(z_trg).squeeze()
        result = Image.fromarray((result * 255).astype(np.uint8))
        results.append(result)

        return results

    def _encode_prompt(
        self,
        prompt,
        trg_prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        trg_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if trg_prompt_embeds is None:
            trg_text_inputs = self.tokenizer(
                trg_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            trg_text_input_ids = trg_text_inputs.input_ids
            trg_untruncated_ids = self.tokenizer(
                trg_prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if trg_untruncated_ids.shape[-1] >= trg_text_input_ids.shape[
                -1
            ] and not torch.equal(trg_text_input_ids, trg_untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    trg_untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
            trg_attention_mask = trg_text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
            trg_attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        trg_prompt_embeds = self.text_encoder(
            trg_text_input_ids.to(device),
            attention_mask=trg_attention_mask,
        )
        trg_prompt_embeds = trg_prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        trg_prompt_embeds = trg_prompt_embeds.to(
            dtype=self.text_encoder.dtype, device=device
        )

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.stack([negative_prompt_embeds, prompt_embeds], axis=1)
            trg_prompt_embeds = torch.stack(
                [negative_prompt_embeds, trg_prompt_embeds], axis=1
            )

        return prompt_embeds, trg_prompt_embeds
