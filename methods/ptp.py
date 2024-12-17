import abc
from typing import Optional, Tuple, List, Union, Dict
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
from tqdm import tqdm
from IPython.display import display
import cv2


LOW_RESOURCE = False

MAX_NUM_WORDS = 77
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5


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


class LocalBlend:
    # LocalBlend类用于实现局部混合编辑，基于注意力图生成局部掩码，从而在图像中应用局部的编辑效果

    def __call__(self, x_t, attention_store):
        k = 1  # 定义用于池化和插值的核大小，控制混合的范围
        # 从attention_store中提取某些层次的注意力图，以便生成局部编辑的掩码
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        # 调整这些注意力图的形状，使它们与alpha_layers匹配，用于后续计算
        maps = [
            item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS)
            for item in maps
        ]
        # 将这些注意力图在指定维度上拼接，以便同时处理多个图层
        maps = torch.cat(maps, dim=1)
        # 将拼接后的注意力图与alpha层（编辑区域的权重）相乘，并对结果进行求和和平均，生成最终的编辑掩码
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        # 使用池化操作对生成的掩码进行处理，确保它在局部区域内平滑
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        # 将掩码插值到与输入图像相同的尺寸，保证它适用于整个图像
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        # 对掩码进行归一化处理，确保其值在适当的范围内
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        # 使用阈值操作对掩码进行二值化，确保只有高于阈值的部分被应用于局部编辑
        mask = mask.gt(self.threshold)
        # 将局部编辑掩码应用到输入图像中，混合输入图像与经过编辑的部分
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t  # 返回局部编辑后的图像

    def __init__(self, pipe, prompts: List[str], words: List[List[str]], threshold=0.3):
        # 初始化LocalBlend类，设置用于局部编辑的alpha层和阈值
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        # 遍历每个文本提示和相关单词，构建用于编辑的alpha层
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                # 根据单词在提示中的位置生成索引，并在alpha层中设置相应位置为1
                ind = get_word_inds(prompt, word, pipe.tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        # 将alpha层和阈值存储为类的属性
        self.alpha_layers = alpha_layers.to(pipe.device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    # AttentionControl类是一个抽象基类，负责控制注意力层的处理流程，提供基本的回调和前向传播功能

    def step_callback(self, x_t):
        # 在每个步骤中执行回调操作，默认直接返回输入
        return x_t

    def between_steps(self):
        # 在每个步骤之间执行的操作，可以由子类实现具体的行为
        return

    @property
    def num_uncond_att_layers(self):
        # 返回无条件（unconditioned）注意力层的数量，在低资源模式下为0
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 抽象方法，负责处理注意力层的前向传播，具体实现由子类提供
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # 在类实例被调用时执行注意力层的处理，控制是否应用无条件注意力层的逻辑
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                # 在低资源模式下，直接处理所有注意力层
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # 如果不是低资源模式，只处理一半的注意力层（提高效率）
                h = attn.shape[0]
                attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1  # 当前处理的注意力层计数增加
        # 如果所有注意力层都处理完，重置计数器并进入下一个步骤
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()  # 处理步骤之间的操作
        return attn  # 返回处理后的注意力图

    def reset(self):
        # 重置当前步骤和注意力层计数器
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        # 初始化注意力控制器的状态
        self.cur_step = 0
        self.num_att_layers = -1  # 注意力层的总数，在具体的实现中会被设置
        self.cur_att_layer = 0  # 当前处理的注意力层


class AttentionStore(AttentionControl):
    # AttentionStore类继承自AttentionControl，用于存储注意力图，方便后续的操作

    @staticmethod
    def get_empty_store():
        # 返回一个空的字典，存储不同层次的注意力图
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 根据是交叉注意力还是自注意力，将注意力图存储在对应的字典键下
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # 避免内存过载，如果注意力图过大则跳过
            self.step_store[key].append(attn)
        return attn  # 返回未修改的attn

    def between_steps(self):
        # 在每一步之间进行操作，将当前步骤存储的注意力图与累积的注意力图合并
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    # 累加每一步的注意力图，便于后续平均计算
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()  # 重置step_store为新的空字典

    def get_average_attention(self):
        # 返回每个层次上平均的注意力图
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        # 重置步骤和存储的注意力图
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        # 初始化AttentionStore，设置空的注意力图存储
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    # AttentionControlEdit类继承自AttentionStore，用于在交叉注意力和自注意力过程中进行编辑控制。

    def step_callback(self, x_t):
        # 如果启用了局部混合，则在每个步骤中对x_t应用局部混合
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t  # 返回经过局部混合处理的x_t

    def replace_self_attention(self, attn_base, att_replace):
        # 替换自注意力，如果注意力图尺寸较小（小于16x16），使用基准注意力扩展替换注意力
        if att_replace.shape[2] <= 16**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace  # 否则，直接返回替换的注意力图

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        # 抽象方法，用于交叉注意力替换，具体逻辑由子类实现
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 前向传播过程中控制注意力替换的逻辑
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (
            self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        ):
            h = attn.shape[0] // (self.batch_size)  # 计算每个批次的注意力层高度
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]  # 将基准注意力图和替换部分分离
            if is_cross:
                # 如果是交叉注意力，则应用alpha权重进行注意力替换
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                attn[1:] = attn_repalce_new  # 更新替换后的注意力
            else:
                # 如果是自注意力，直接替换
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(
                self.batch_size * h, *attn.shape[2:]
            )  # 重新调整注意力图形状
        return attn  # 返回处理后的注意力图

    def __init__(
        self,
        pipe,
        prompts,  # 提示文本列表
        num_steps: int,  # 生成过程的总步数
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Tuple[float, float]]
        ],  # 交叉注意力替换的步数
        self_replace_steps: Union[float, Tuple[float, float]],  # 自注意力替换的步数
        local_blend: Optional[LocalBlend],  # 局部混合对象
    ):
        # 初始化方法，设置注意力替换相关的参数
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)  # 设置批次大小为提示文本的数量
        self.cross_replace_alpha = get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, pipe.tokenizer
        ).to(
            pipe.device
        )  # 获取用于交叉注意力替换的alpha权重
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps  # 如果是浮点数，转为元组
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(
            num_steps * self_replace_steps[1]
        )  # 计算自注意力替换的步数范围
        self.local_blend = local_blend  # 设置局部混合


def aggregate_attention(
    prompts: List[str],
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    is_cross: bool,
    select: int,
):
    # 聚合注意力图，用于将多层的注意力图合并成一个整体
    out = []  # 存储每层注意力图的输出
    attention_maps = (
        attention_store.get_average_attention()
    )  # 从AttentionStore中获取平均注意力图
    num_pixels = res**2  # 计算图像的像素数（res表示图像的分辨率，res^2表示总像素数）

    for location in from_where:  # 遍历指定位置（例如down_cross或up_self等注意力层）
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            # 遍历注意力图，如果图像大小与num_pixels匹配，进行处理
            if item.shape[1] == num_pixels:
                # 重新调整注意力图的形状，使其与分辨率和选择的提示词匹配
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[
                    select
                ]
                out.append(cross_maps)  # 将处理后的结果添加到输出列表

    out = torch.cat(out, dim=0)  # 将所有处理的注意力图沿着第一个维度拼接
    out = out.sum(0) / out.shape[0]  # 对所有拼接的注意力图求平均
    return out.cpu()  # 返回计算出的注意力图，并将其转换为CPU上的张量


def show_cross_attention(
    pipe,
    prompts: List[str],
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    select: int = 0,
):
    # 显示交叉注意力图，展示每个token在图像中的注意力权重
    tokens = pipe.tokenizer.encode(prompts[select])  # 对提示词进行编码，生成token序列
    decoder = pipe.tokenizer.decode  # 获取解码器，将token转为人类可读的文本
    attention_maps = aggregate_attention(
        prompts, attention_store, res, from_where, True, select
    )  # 聚合交叉注意力图
    images = []  # 用于存储生成的图像

    for i in range(len(tokens)):  # 遍历每个token
        image = attention_maps[:, :, i]  # 提取与当前token对应的注意力图
        image = 255 * image / image.max()  # 将注意力图归一化到0-255之间
        image = image.unsqueeze(-1).expand(
            *image.shape, 3
        )  # 将单通道的注意力图扩展为三通道RGB图像
        image = image.numpy().astype(np.uint8)  # 转换为numpy数组，并转换为uint8类型
        image = np.array(
            Image.fromarray(image).resize((256, 256))
        )  # 调整图像大小为256x256
        image = text_under_image(
            image, decoder(int(tokens[i]))
        )  # 在图像下方添加对应的token文本
        images.append(image)  # 将图像添加到列表中

    view_images(np.stack(images, axis=0))  # 将所有图像沿第一个维度堆叠，并显示


def show_self_attention_comp(
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    max_com=10,
    select: int = 0,
):
    # 显示自注意力主成分分析图，展示前几个主成分的注意力分布
    attention_maps = (
        aggregate_attention(attention_store, res, from_where, False, select)
        .numpy()
        .reshape((res**2, res**2))
    )
    # 聚合自注意力图，并将其转为二维矩阵，表示像素间的关系
    u, s, vh = np.linalg.svd(
        attention_maps - np.mean(attention_maps, axis=1, keepdims=True)
    )  # 对注意力图进行SVD分解
    images = []  # 存储生成的主成分图像

    for i in range(max_com):  # 取前max_com个主成分
        image = vh[i].reshape(res, res)  # 将主成分重新调整为原图的分辨率
        image = image - image.min()  # 将主成分的最小值调整为0
        image = 255 * image / image.max()  # 将主成分归一化到0-255之间
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(
            np.uint8
        )  # 将单通道扩展为三通道
        image = Image.fromarray(image).resize((256, 256))  # 将图像调整为256x256大小
        image = np.array(image)  # 转换为numpy数组
        images.append(image)  # 将处理好的图像添加到列表

    view_images(np.concatenate(images, axis=1))  # 将所有主成分图像拼接并显示


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
    pipe,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.0,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(pipe, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = pipe.tokenizer(
        [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = pipe.bert(uncond_input.input_ids.to(pipe.device))[0]

    text_input = pipe.tokenizer(
        prompt, padding="max_length", max_length=77, return_tensors="pt"
    )
    text_embeddings = pipe.bert(text_input.input_ids.to(pipe.device))[0]
    latent, latents = init_latent(latent, pipe, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    pipe.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(pipe.scheduler.timesteps):
        latents = diffusion_step(pipe, controller, latents, context, t, guidance_scale)

    image = latent2image(pipe.vqvae, latents)

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


def run_and_display(
    pipe, prompts, controller, latent=None, run_baseline=False, generator=None
):
    # 运行图像生成并显示结果
    # prompts: 文本提示列表，用于引导图像生成
    # controller: 注意力控制器，用于控制生成过程中的注意力机制
    # latent: 初始潜在变量，用于调整生成结果（可选）
    # run_baseline: 是否运行基线模型，比较有无Prompt-to-Prompt方法的生成效果
    # generator: 随机数生成器，用于生成不同的图像（可选）

    if run_baseline:  # 如果设置了运行基线
        print("w.o. prompt-to-prompt")  # 打印不使用Prompt-to-Prompt的方法
        # 递归调用run_and_display，但使用空的控制器（不进行Prompt-to-Prompt修改）
        images, latent = run_and_display(
            pipe,
            prompts,
            EmptyControl(),
            latent=latent,
            run_baseline=False,
            generator=generator,
        )
        print("with prompt-to-prompt")  # 打印使用Prompt-to-Prompt的方法

    # 使用ptp_utils的text2image_ldm_stable方法，通过提示文本生成图像
    # ldm_stable是稳定扩散模型，controller控制注意力，latent用于生成潜在空间
    # num_inference_steps是生成过程的步数，guidance_scale用于引导生成的强度
    # low_resource表示是否使用低资源模式，节约内存
    images, x_t = text2image_ldm_stable(
        pipe,
        prompts,
        controller,
        latent=latent,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        low_resource=LOW_RESOURCE,
    )

    # 显示生成的图像
    view_images(images)

    # 返回生成的图像以及潜在变量
    return images, x_t


class AttentionReplace(AttentionControlEdit):
    # AttentionReplace类继承自AttentionControlEdit，用于替换交叉注意力

    def replace_cross_attention(self, attn_base, att_replace):
        # 使用爱因斯坦求和约定，将基准注意力和替换注意力进行相乘组合
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        pipe,  # 用于分词的tokenizer
        prompts,  # 提示文本
        num_steps: int,  # 生成过程的步数
        cross_replace_steps: float,  # 交叉注意力替换步数
        self_replace_steps: float,  # 自注意力替换步数
        local_blend: Optional[LocalBlend] = None,  # 局部混合（可选）
    ):
        super(AttentionReplace, self).__init__(
            pipe,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        # 获取用于交叉注意力替换的映射器
        self.mapper = get_replacement_mapper(prompts, pipe.tokenizer).to(pipe.device)


class AttentionRefine(AttentionControlEdit):
    # AttentionRefine类用于精细化处理交叉注意力替换，结合alphas权重

    def replace_cross_attention(self, attn_base, att_replace):
        # 通过映射器获取基准注意力的替换部分，并通过alphas进行加权组合
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace  # 返回替换后的注意力图

    def __init__(
        self,
        pipe,
        prompts,  # 提示文本
        num_steps: int,  # 生成步骤
        cross_replace_steps: float,  # 交叉注意力替换步数
        self_replace_steps: float,  # 自注意力替换步数
        local_blend: Optional[LocalBlend] = None,  # 局部混合（可选）
    ):
        super(AttentionRefine, self).__init__(
            pipe,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        # 获取映射器和alphas权重，用于精细化交叉注意力替换
        self.mapper, alphas = get_refinement_mapper(prompts, pipe.tokenizer)
        self.mapper, alphas = self.mapper.to(pipe.device), alphas.to(pipe.device)
        self.alphas = alphas.reshape(
            alphas.shape[0], 1, 1, alphas.shape[1]
        )  # 调整alphas的形状


class AttentionReweight(AttentionControlEdit):
    # AttentionReweight类用于重新加权交叉注意力，结合之前的控制器输出

    def replace_cross_attention(self, attn_base, att_replace):
        # 如果有之前的控制器，使用其输出作为基准注意力进行替换
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(
                attn_base, att_replace
            )
        # 将基准注意力与equalizer进行加权组合
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace  # 返回加权后的注意力图

    def __init__(
        self,
        pipe,
        prompts,  # 提示文本
        num_steps: int,  # 生成过程的步数
        cross_replace_steps: float,  # 交叉注意力替换步数
        self_replace_steps: float,  # 自注意力替换步数
        equalizer,  # 权重调整器
        local_blend: Optional[LocalBlend] = None,  # 局部混合（可选）
        controller: Optional[AttentionControlEdit] = None,  # 前一个控制器（可选）
    ):
        super(AttentionReweight, self).__init__(
            pipe,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        # 初始化equalizer和前一个控制器
        self.equalizer = equalizer.to(pipe.device)
        self.prev_controller = controller  # 设置之前的控制器，便于多次加权


def get_equalizer(
    pipe,
    text: str,  # 输入文本
    word_select: Union[int, Tuple[int, ...]],  # 选中的单词索引
    values: Union[List[float], Tuple[float, ...]],  # 加权值列表
):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)  # 如果是整数或字符串，转换为元组
    equalizer = torch.ones(len(values), 77)  # 初始化一个全为1的张量
    values = torch.tensor(values, dtype=torch.float32)  # 将值转换为浮点数张量
    for word in word_select:
        # 获取文本中选定单词的索引
        inds = get_word_inds(text, word, pipe.tokenizer)
        equalizer[:, inds] = values  # 将选定单词的权重调整为指定值
    return equalizer  # 返回构造的equalizer


class EmptyControl(AttentionControl):
    # EmptyControl类是一个空的注意力控制器，不改变任何注意力图，仅返回原始的attn

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 返回原始的attn，没有任何修改
        return attn
