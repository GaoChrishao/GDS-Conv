import torch
import torch.nn as nn
import numpy as np

from typing import List

from fairseq.modules.activations import Swish
from fairseq.modules.layer_norm import LayerNorm

from matplotlib import pyplot as plt
from fairseq.modules.gmm import GaussianMixture1, Shrink
from fairseq.modules.utils import print_save_mask_info


def get_activation_class(activation: str, dim=None):
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "glu":
        assert dim is not None
        return nn.GLU(dim=dim)
    elif activation == "swish":
        return Swish()
    elif activation == "none":
        return nn.Identity()
    else:
        raise RuntimeError("activation function {} not supported".format(activation))


class TransposeLast(nn.Module):

    @staticmethod
    def forward(x):
        return x.transpose(-1, -2).contiguous()


def get_norm(norm_type, size, transpose=False):
    trans = nn.Identity()
    if transpose:
        trans = TransposeLast()
    if norm_type == "batch1d":
        return nn.Sequential(trans, nn.BatchNorm1d(size), trans)
    elif norm_type == "batch2d":
        return nn.Sequential(trans, nn.BatchNorm2d(size), trans)
    elif norm_type == "layer":
        return nn.Sequential(trans, LayerNorm(size), trans)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise RuntimeError("normalization type {} not supported".format(norm_type))


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            out_channels: int,
            kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        inner_x = []
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
            inner_x.append(x)
        _, _, out_seq_len = x.size()
        # x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        out_inner_x = []
        for x in inner_x:
            out_inner_x.append(x.transpose(1, 2).transpose(0, 1).contiguous())
        return out_inner_x, self.get_out_seq_lens_tensor(src_lengths)


# fairseq style
class Conv1dSubsampling(nn.Module):
    """Conv1d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        in_dim: input feature dimension
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_length, in_dim)
        Output: (batch_size, out_length, out_dim)

    """

    def __init__(self, num_layers,
                 in_dim, filters, kernel_size, stride=2,
                 norm="none", act="glu"):
        super(Conv1dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch1d", "layer", "none"]
        assert act in ["relu", "swish", "glu", "none"]

        # Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_dim if layer_id == 0 else filters[layer_id - 1] // 2 if act == "glu" else filters[layer_id - 1],
                      filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                      kernel_size,
                      stride=stride,
                      padding=(kernel_size - 1) // 2),
            get_norm(norm,
                     filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                     transpose=True if norm == "layer" else False),
            get_activation_class(act, dim=1)
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (T, B, D) -> (B, D, T)
        x = x.permute(1, 2, 0)
        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1

        # (B, D, T) -> (T, B, D)
        x = x.permute(2, 0, 1)
        return x, x_len


class Conv2dSubsampling(nn.Module):
    """Conv2d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (in_length, batch_size in_dim)
        Output: (out_length, batch_size, out_dim)

    """

    def __init__(self, num_layers,
                 in_dim, filters, kernel_size, stride=2,
                 norm="none", act="glu"):
        super(Conv2dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch2d", "none"]
        assert act in ["relu", "swish", "glu", "none"]

        # Conv 2D Subsampling Layers

        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1] // 2 if act == "glu" else filters[layer_id - 1],
                      filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                      kernel_size,
                      stride=stride,
                      padding=(kernel_size - 1) // 2),
            get_norm(norm,
                     filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                     transpose=True if norm == "layer" else False),
            get_activation_class(act, dim=1)
        ) for layer_id in range(num_layers)])
        self.linear = nn.Linear(filters[-1] * in_dim // 2 ** num_layers, filters[-1])

    def forward(self, x, x_len):

        # (T, B, D) -> (B, D, T) -> (B, 1, D, T)
        x = x.permute(1, 2, 0).unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length).permute(2, 0, 1)
        x = self.linear(x)

        return x, x_len


# v1
def mask_pad(raw_mask, lens: torch.LongTensor) -> torch.BoolTensor:
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    pad_mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    pad_mask = pad_mask.expand(bsz, -1) < lens.view(bsz, 1).expand(-1, max_lens)

    mask = raw_mask & pad_mask

    # 做个异常处理，防止padding后的mask全为0，进而防止句子为0
    # 这种异常处理，暂时也有问题：异常的句子，会将mask全部设置为1,忽略了padding，但是实践发现几乎不存在这种情况
    error_mask = (~mask).all(-1).unsqueeze(1)
    mask = mask | error_mask

    return mask


# v2
# 根据给定的mask，进行padding后，计算长度，再抽取
def extrac_sentence_by_mask(pre_lens, mask, x, bsz):
    # 根据pre_lens,将mask中的padding位置设置为False
    mask = mask_pad(mask, pre_lens)

    # 融合后的新长度
    new_out_lens = mask.long().sum(-1)

    # 异常处理
    cat_lens = torch.cat((new_out_lens, pre_lens)).reshape(2, -1)
    new_out_lens, indices = torch.min(cat_lens, dim=0)

    # 异常处理的句子数量
    error_sentence_num = indices.sum()
    if error_sentence_num > 0:
        print("gch:异常句子数量:{}".format(error_sentence_num.item()))

    max_len = new_out_lens.max()

    sentences = x[mask, :]

    out = x.new_zeros((bsz, max_len, x.size(-1)))

    tmp_1 = 0
    tmp_2 = 0
    for i in range(bsz):
        tmp_2 += new_out_lens[i].item()
        out[i, : new_out_lens[i]] = sentences[tmp_1:tmp_2]
        tmp_1 += new_out_lens[i].item()

    new_out_lens = new_out_lens.long()
    return out, new_out_lens



pool_need_unsqueeze = ("1.8" in torch.__version__)


def avg_pool(mask, length):
    if pool_need_unsqueeze:
        mask_low = nn.functional.adaptive_avg_pool1d(mask.unsqueeze(0), length).squeeze(0)
    else:
        mask_low = nn.functional.adaptive_avg_pool1d(mask, length)
    # mask_low = nn.functional.adaptive_avg_pool1d(mask.unsqueeze(0), length).squeeze(0)
    return mask_low


def print_save_mask_info(old_mask, src_tokens, new_mask, all_shrink_ratio, current_shrink_ratio, count, tag="mask",
                         save_pic=True):
    old_mask_ratio = old_mask[0].long().sum(-1).item() / old_mask.size(-1)
    print("mask高能量比重:{:.3f}".format(old_mask_ratio))
    print("当前压缩比例:{:.3f} 总压缩比例:{:.3f}".format(current_shrink_ratio, all_shrink_ratio))

    if not save_pic:
        return

    if new_mask == None:
        return

    fig = plt.figure(dpi=300, figsize=(4, 3), constrained_layout=True)
    ax = fig.subplots(3, 1, sharex=True)

    # print(src_tokens[0,:10])
    tmp1 = src_tokens[0].cpu().T
    tmp2 = old_mask[0].float().cpu().repeat(80, 1)

    tmp1 = tmp1.to(torch.float32).numpy()
    tmp2 = tmp2.to(torch.float32).numpy()

    ax[0].imshow(tmp1, interpolation='nearest', aspect='auto')

    # ax[0].plot(old_mask[0].float().cpu().repeat(80,1).numpy(), alpha=0.5)
    ax[1].imshow(tmp1, interpolation='nearest', aspect='auto')
    ax[1].imshow(tmp2, interpolation='nearest', aspect='auto', alpha=0.2, cmap="gray")

    # print(old_mask[0,:10])
    new_mask = new_mask.to(torch.float32)
    tmp3 = avg_pool(new_mask, old_mask.shape[-1])[0].float().cpu().repeat(80, 1)
    # print(tmp3.shape)
    tmp3 = tmp3.to(torch.float32).numpy()

    ax[2].imshow(tmp1, interpolation='nearest', aspect='auto')
    ax[2].imshow(tmp3, interpolation='nearest', aspect='auto', alpha=0.2, cmap="gray")

    # print(new_mask[0,10])

    # ax[1].imshow(src_tokens[0].cpu().T.numpy(), interpolation='nearest', aspect='auto')
    # ax[1].plot(nn.functional.adaptive_avg_pool1d(new_mask.float(), old_mask.shape[-1])[0].float().cpu().numpy())

    plt.savefig("/home/gaochenghao/pic/{}_{}".format(tag, count))
    plt.clf()
    plt.cla()
    plt.close("all")



# 2022/09/25
def combine_two_stride_v5(x_low, x_height, mask, pool_thresh,  pre_lens,  false_bool,batch_first_out=False):
    bsz, hidden_size, seq_length_low = x_low.size()

    # B x D x T = > B X T X D
    x_height = x_height.transpose(1, 2)
    x_low = x_low.transpose(1, 2)

    #mask长度与x_low对齐，使用pool的方式缩短mask的长度
    mask = mask.float()
    mask_low = avg_pool(mask, seq_length_low)


    # 对mask进行二值化，并转换成bool,True为能量高的地方
    mask_low = torch.where(mask_low > pool_thresh, 1, 0)
    mask_low = mask_low.bool()

    mask_height = ~mask_low

    # mask_height_in_low_pos ==> mask_height
    # mask_height：x_height中，需要融合进入x_low的元素
    mask_height_need_pos = mask_height[:, false_bool[:seq_length_low]]

    # 将mask_reverse的奇数位置设置为False，这样就能得到x_height中需要融合的元素对应在x_low中的位置
    mask_height_in_low_pos = mask_height & false_bool[:seq_length_low]

    # 将x_height中需要融合的元素 放入 对应位置的x_low中
    # 如果注释掉一下这句代码，BLEU从21.90掉到21.72,相当于去除一部分信息
    # 在2x2与2x4进行融合时，由于2x4的数值与2x2奇数位置一样，以下代码的融合将不生效
    x_height_need_1D = x_height.masked_select(mask_height_need_pos.unsqueeze(-1)).reshape(-1, hidden_size)
    x_low_1D = x_low.reshape(-1, hidden_size)
    map_index = torch.arange(bsz * seq_length_low).reshape(bsz, -1).to(x_low.device)
    map_index = map_index.masked_select(mask_height_in_low_pos).reshape(-1)
    map_index = map_index.unsqueeze(-1).repeat(1, hidden_size)

    x_low_1D.scatter_(dim=0, index=map_index, src=x_height_need_1D)

    x_low = x_low_1D.reshape(bsz, seq_length_low, hidden_size)

    #x_low[mask_height_in_low_pos, :] = x_height[mask_height_need_pos, :]



    # 做or运算，mask_low_need为原先的高能量区域，mask_reverse为从mask_height中融入到mask_low的低能量区域
    final_mask = (mask_height_in_low_pos | mask_low)  # 做 or 运算

    # 根据pre_lens,将final_mask中的padding位置设置为False
    final_mask = mask_pad(final_mask,pre_lens)

    #融合后的新长度
    new_out_lens = final_mask.long().sum(-1)

    max_len = new_out_lens.max()
    sentences = x_low[final_mask, :]

    # 使用scatter，进行映射，再reshape
    out_1D=x_low.new_zeros((bsz*max_len, x_low.size(-1)))


    #以上max_index的生成非常慢，需要使用非遍历手段，或者尝试使用cpu完成？
    map_index = np.arange(sentences.size(0))
    sum_tmp=0
    sum_shift=0
    max_len_np=max_len.cpu().numpy()
    new_out_lens_np=new_out_lens.cpu().numpy()
    for i in range(bsz):
        map_index[sum_tmp:sum_tmp+new_out_lens_np[i]]+=sum_shift
        sum_tmp+=new_out_lens_np[i]
        sum_shift+=(max_len_np-new_out_lens_np[i])
    map_index=torch.from_numpy(map_index).to(x_low.device)

    out_1D.scatter_(0,map_index.unsqueeze(1).repeat(1,hidden_size),sentences)

    out = out_1D.reshape(bsz, max_len, hidden_size)

    if not batch_first_out:
        out = out.transpose(0,1).contiguous()

    return out, new_out_lens.long(), mask_low



# version: 0.11
# 2022/10/18
class Conv1dSubsamplingShrink(nn.Module):
    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            out_channels: int,
            kernel_sizes: List[int] = (3, 3),
            debug: bool = False,
            method: str = "None",
            shrink_value: List[float] = (0.5),
            random_mask=False,
            thresh_dynamic=False,
            pool_thresh=False,
            gmm_iter=100,
            gmm_iter_per=10,
            gmm_prob=75,
            gmm_num=2,
            gmm_dim=80,
    ):
        super(Conv1dSubsamplingShrink, self).__init__()

        self.n_layers = len(kernel_sizes)
        self.convs = [
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,

            )
            for i, k in enumerate(kernel_sizes)
        ]

        if "no_shrink_8x_new" in method:
            k = kernel_sizes[-1]
            self.kernel_sizes.append(k)
            self.convs.append(nn.Conv1d(out_channels,out_channels*2,k,stride=k,padding=k//2))


        self.conv_layers = nn.ModuleList(
            [conv for conv in self.convs]
        )

        self.false_bool = None

        self.debug = debug
        self.all_ratio = 0
        self.all_shrink_length = 0
        self.all_pre_shrink_length = 0
        self.count = 0

        self.mask_generate = None
        self.shrink_value = shrink_value
        self.random_mask = random_mask
        self.thresh_dynamic = thresh_dynamic
        self.pool_thresh = pool_thresh

        self.method = method
        if "thresh" in self.method or "mean" in self.method:
            self.mask_generate = Shrink(debug=False, mask_value=shrink_value,
                                        feature_sample=40,
                                        random_mask=random_mask,
                                        thresh_dynamic=thresh_dynamic,
                                        method=self.method)
        elif "gmm" in self.method:
            # max_iter越大，判断出的高能量mask越多，压缩效果越弱
            self.mask_generate = GaussianMixture1(gmm_num, gmm_dim, covariance_type="diag", max_iter=gmm_iter,iter_per=gmm_iter_per,gmm_prob=gmm_prob)
        elif "no_shrink" in self.method:
            self.mask_generate =None
        else:
            raise RuntimeError("Subsampling type {} not supported".format(method))

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor, strides=[2, 2]):
        out = in_seq_lens_tensor.clone()
        for i in range(self.n_layers):
            out = ((out.float() - 1) / strides[i] + 1).floor().long()
        return out

    def init_false_bool(self, src_tokens):
        if self.false_bool is None:
            # TF TF TF TF ......
            self.false_bool = torch.arange(2000, requires_grad=False).to(src_tokens.device)
            self.false_bool = (self.false_bool % 2 == 0)

    def show_result(self, new_out_lens, pre_lens, old_mask, src_tokens, mask_low, save_pic=False,method=""):
        shrink_ratio = (new_out_lens.sum(-1) / pre_lens.sum(-1)).item()

        self.all_shrink_length += new_out_lens.sum(-1).item()
        self.all_pre_shrink_length += pre_lens.sum(-1).item()
        all_shrink_ratio = self.all_shrink_length / self.all_pre_shrink_length
        self.count += 1

        # 训练或解码的前100个step,每10个step输出下压缩率和图片
        # if self.debug and self.count <= 1000 and self.count % 100 == 0:
        #     print_save_mask_info(old_mask, src_tokens, mask_low, all_shrink_ratio, shrink_ratio, self.count,
        #                          tag=self.method, save_pic=save_pic)
        #
        # 训练时每1000个step，输出下压缩率
        if self.training and self.count % 500 == 0:
            print_save_mask_info(old_mask, src_tokens, mask_low, all_shrink_ratio, shrink_ratio, self.count,
                                 tag=self.method, save_pic=True)

            # 每2171个step重置下压缩率计算数据
            # de 1965
            # fr 2384
            if self.count % 500 == 0:
                self.all_shrink_length = 0
                self.all_pre_shrink_length = 0

        # 训练中，校验时输出
        if not self.training and self.count%10==0:
            print_save_mask_info(old_mask, src_tokens, mask_low, all_shrink_ratio, shrink_ratio, self.count,
                                 tag=self.method.format(self.method), save_pic=False)

        # 测试test-other的IM效果
        # if not self.training:
        #     print_save_mask_info(old_mask, src_tokens, mask_low, all_shrink_ratio, shrink_ratio, self.count,
        #                          tag=self.method.format(self.method), save_pic=True)

        # 测试注意力机制
        # if self.debug and not self.training and self.count <= 500 and self.count%10==0:
        #     print_save_mask_info(old_mask, src_tokens, mask_low, all_shrink_ratio, shrink_ratio, self.count,
        #                          tag="var", save_pic=True)

        # if self.count%10==0:
        #     print_save_mask_info(old_mask, src_tokens, mask_low, all_shrink_ratio, shrink_ratio, self.count,
        #                          tag="{}_{}".format(self.method,"1w_90I1"),save_pic=False)

        # if self.training and self.count % 100==0:
        #     print(self.mask_generate.mu.data)
        # print(self.mask_generate.mu.data[0,1,:20].item())

    def forward(self, src_tokens, src_lengths=None, *args):
        self.init_false_bool(src_tokens)

        # Fairseq-S2T
        in_seq_len, bsz, dim = src_tokens.size()  # (T, B, D)
        #(T, B, D) -> (B, T, D)
        src_tokens = src_tokens.transpose(0, 1)

        # MSP-ST
        #bsz, in_seq_len, dim = src_tokens.size()  # B x T x (C x D)

        mask = None
        if self.mask_generate is not None:
            mask = self.mask_generate.predict(src_tokens)
            if type(mask) == list:
                mask = mask[0]

        mask_low = None
        old_mask = mask
        # mask: B X T1

        random_mask = (self.training and self.random_mask)
        if random_mask:
            tmp = torch.rand(mask.shape[0], mask.shape[1], device=src_tokens.device)
            mask_random = tmp > 0.1
            mask = mask & mask_random
        if "no_shrink_4x" in self.method:
            # (B, T, D) -> (B, D, T)
            x = src_tokens.transpose(1, 2)
            for i in range(self.n_layers):
                conv = self.conv_layers[i]
                x = conv(x)
                x = nn.functional.glu(x, dim=1)
            out = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
            new_out_lens=self.get_out_seq_lens_tensor(src_lengths)
        elif "no_shrink_8x_new" in self.method:
            # (B, T, D) -> (B, D, T)
            x = src_tokens.transpose(1, 2)
            for i in range(self.n_layers):
                conv = self.conv_layers[i]
                x = conv(x)
                x = nn.functional.glu(x, dim=1)
            out = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
            new_out_lens=self.get_out_seq_lens_tensor(src_lengths,strides=[2,2,2])
        elif "no_shrink_8x" in self.method:
            # (B, T, D) -> (B, D, T)
            self.convs[0].stride = 4
            self.convs[1].stride = 2
            x = src_tokens.transpose(1, 2)
            for i in range(self.n_layers):
                conv = self.conv_layers[i]
                x = conv(x)
                x = nn.functional.glu(x, dim=1)
            out = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
            new_out_lens=self.get_out_seq_lens_tensor(src_lengths,strides=[4,2])
        # 下采样后去除
        elif "drop1" in self.method:
            # (B, T, D) -> (B, D, T)
            x = src_tokens.transpose(1, 2)
            for i in range(self.n_layers):
                conv = self.conv_layers[i]
                x = conv(x)
                x = nn.functional.glu(x, dim=1)

            # 正常的pool，减少高能量mask的比重，进而提高压缩率
            # mask长度与x对齐，使用pool的方式缩短mask的长度
            mask = mask.float()
            mask = avg_pool(mask, x.shape[-1])
            # 对mask进行二值化，并转换成bool,True为能量高的地方
            mask = torch.where(mask > self.pool_thresh, 1, 0)
            mask = mask.bool()
            # mask:[B X T]

            # 原先采用4倍压缩获得的长度
            pre_lens = self.get_out_seq_lens_tensor(src_lengths, [2, 2])

            #  (B, D, T) -> (B, T, D)
            x = x.transpose(1, 2)
            out, new_out_lens = extrac_sentence_by_mask(pre_lens, mask, x, bsz)

            # (B, T, D) -> (T, B, D)
            out = out.transpose(0, 1)
            mask_low = mask
            self.show_result(new_out_lens, pre_lens, old_mask, src_tokens, mask_low, False)

        # 下采样前去除
        elif "drop" in self.method:
            # (B, T, D)
            x = src_tokens

            # 按照mask去掉src_tokens中的指定位置，并重新计算长度
            out, new_src_length = extrac_sentence_by_mask(src_lengths, mask, x, bsz)

            x = out.transpose(1, 2).contiguous()  # -> B x (C x D) x T

            for i in range(self.n_layers):
                conv = self.conv_layers[i]
                x = conv(x)
                x = nn.functional.glu(x, dim=1)

            # (B, D, T) -> (T, B, D)
            out = x.permute(2, 0, 1)

            new_out_lens = self.get_out_seq_lens_tensor(new_src_length, [2, 2])

            # 原先采用4倍压缩获得的长度
            pre_lens = self.get_out_seq_lens_tensor(src_lengths, [2, 2])
            self.show_result(new_out_lens, pre_lens, old_mask, src_tokens, mask_low, False)

        # 默认: 步长融合
        else:

            src_tokens_tmp = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T

            # x_low为小步长
            x_low = src_tokens_tmp

            # 小步长
            self.convs[0].stride = 2
            self.convs[1].stride = 2
            for i in range(self.n_layers):
                conv = self.conv_layers[i]
                x_low = conv(x_low)
                x_low = nn.functional.glu(x_low, dim=1)

            # 大步长
            x_height = src_tokens_tmp
            # with torch.no_grad():
            self.convs[0].stride = 4
            self.convs[1].stride = 2
            for i in range(self.n_layers):
                conv = self.conv_layers[i]
                x_height = conv(x_height)
                x_height = nn.functional.glu(x_height, dim=1)

            pre_lens = self.get_out_seq_lens_tensor(src_lengths, [2, 2])

            # (B, D, T) -> (T, B, D)
            out, new_out_lens, mask_low = combine_two_stride_v5(x_low, x_height, mask, self.pool_thresh,
                                                                pre_lens,
                                                                self.false_bool)

            # new_out_lens[:]=new_out_lens.max()
            # print("shrink_ratio_05:{:.2f}".format(out.size(0)/x_low.size(-1)))

            # 随机比例
            # ratio = random.random()/5+0.8
            # shrink_length = math.floor(out.size(0) * ratio)
            # out = out[:shrink_length,:, : ]
            # new_out_lens = torch.floor(new_out_lens * ratio).long()

            # new_out_lens[:]=new_out_lens.max()

            # (B, D, T) -> (T, B, D)
            # 以下两行测试使用

            # ratio=0.9
            # shrink_length = math.floor(x_low.size(-1) * ratio)
            # #x_low = x_low[:, :, :shrink_length]
            #
            # #new_out_lens = torch.floor(self.get_out_seq_lens_tensor(src_lengths) * ratio).long()
            # #new_out_lens[:]=new_out_lens.max()

            # max_index = torch.argmax(new_out_lens)
            # max_len=new_out_lens.max().item()

            # random_len = torch.randint(int(max_len*0.8),max_len,new_out_lens.size()).to(new_out_lens.device)
            # random_len[max_index]=new_out_lens[max_index]
            # new_out_lens=random_len

            # out = x_low.permute(2, 0, 1).contiguous()
            # new_out_lens = self.get_out_seq_lens_tensor(src_lengths, [2, 2])

            self.show_result(new_out_lens,pre_lens,old_mask,src_tokens,mask_low,True,self.method)

        return out, new_out_lens




def subsampling(args, out_dim=None):
    subsampling_type = getattr(args, "subsampling_type", "conv1d")
    layers = getattr(args, "subsampling_layers", 2)
    in_dim = args.input_feat_per_channel * args.input_channels
    filters = [getattr(args, "subsampling_filter")] + [args.encoder_embed_dim if out_dim is None else out_dim]
    kernel_size = getattr(args, "subsampling_kernel", 5)
    stride = getattr(args, "subsampling_stride", 2)
    norm = getattr(args, "subsampling_norm", "none")
    activation = getattr(args, "subsampling_activation", "none")

    random_mask = args.random_mask
    thresh_dynamic = args.thresh_dynamic
    pool_thresh = args.pool_thresh
    shrink_value = []
    gmm_iter = args.gmm_iter
    gmm_iter_per = args.gmm_iter_per
    gmm_prob = args.gmm_prob
    gmm_num=args.gmm_num
    gmm_dim=args.gmm_dim

    for value in args.shrink_value.split(","):
        shrink_value.append(float(value))

    # print("----------临时测试----------")
    # self.shrink_value=[0.0,0.4]

    # self.shrink_method = "thresh"
    # self.shrink_value = [0.2]
    # self.thresh_dynamic=True
    # pool_thresh = 0.0
    args.decode_debug = True
    #subsampling_type="no_shrink_4x"

    #测试
    # if "100I10" in subsampling_type:
    #     pool_thresh=0.5

    print("subsampling_type:{}".format(subsampling_type))
    print("shrink-value:{}".format(shrink_value))
    print("random-mask:{}".format(random_mask))
    print("thresh-dynamic:{}".format(thresh_dynamic))
    print("pool-thresh:{}".format(pool_thresh))
    print("gmm_iter:{}".format(gmm_iter))
    print("gmm_iter_per:{}".format(gmm_iter_per))
    print("gmm_prob:{}".format(gmm_prob))
    print("gmm_num:{}".format(gmm_num))
    print("gmm_dim:{}".format(gmm_dim))

    if subsampling_type == "conv1d":
        return Conv1dSubsampling(layers, in_dim, filters, kernel_size, stride, norm, activation)
    elif subsampling_type == "conv2d":
        return Conv2dSubsampling(layers, in_dim, filters, kernel_size, stride, norm, activation)
    else:
        return Conv1dSubsamplingShrink(in_dim,
                                       filters[0],
                                       filters[1],
                                       [int(k) for k in "5,5".split(",")],
                                       args.decode_debug,
                                       subsampling_type,
                                       shrink_value,
                                       random_mask,
                                       thresh_dynamic,
                                       pool_thresh,
                                       gmm_iter,
                                       gmm_iter_per,
                                       gmm_prob,
                                       gmm_num,
                                       gmm_dim
                                       )
