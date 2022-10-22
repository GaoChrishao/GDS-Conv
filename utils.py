import matplotlib
import torch
from torch import nn as nn

import matplotlib.pyplot as plt
matplotlib.use('Agg')


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Permute3D(nn.Module):

    def __init__(self, dim0, dim1, dim2):
        super(Permute3D, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.permute(self.dim0, self.dim1, self.dim2)


pool_need_unsqueeze = ("1.8" in torch.__version__)


def avg_pool(mask, length):
    if pool_need_unsqueeze:
        mask_low = nn.functional.adaptive_avg_pool1d(mask.unsqueeze(0), length).squeeze(0)
    else:
        mask_low = nn.functional.adaptive_avg_pool1d(mask, length)
    # mask_low = nn.functional.adaptive_avg_pool1d(mask.unsqueeze(0), length).squeeze(0)
    return mask_low
def print_save_mask_info(old_mask, src_tokens, new_mask, all_shrink_ratio, current_shrink_ratio, count, tag="mask",save_pic=True):

    old_mask_ratio = old_mask[0].long().sum(-1).item() / old_mask.size(-1)
    print("mask高能量比重:{:.3f}".format(old_mask_ratio))
    print("当前压缩比例:{:.3f} 总压缩比例:{:.3f}".format(current_shrink_ratio, all_shrink_ratio))

    if not save_pic:
        return

    fig = plt.figure(dpi=300, figsize=(4, 3), constrained_layout=True)
    ax = fig.subplots(3, 1, sharex=True)

    #print(src_tokens[0,:10])
    tmp1=src_tokens[0].cpu().T
    tmp2=old_mask[0].float().cpu().repeat(80, 1)

    tmp1=tmp1.to(torch.float32).numpy()
    tmp2=tmp2.to(torch.float32).numpy()


    ax[0].imshow(tmp1, interpolation='nearest', aspect='auto')

    # ax[0].plot(old_mask[0].float().cpu().repeat(80,1).numpy(), alpha=0.5)
    ax[1].imshow(tmp1, interpolation='nearest', aspect='auto')
    ax[1].imshow(tmp2, interpolation='nearest', aspect='auto', alpha=0.2,cmap="gray")

    #print(old_mask[0,:10])
    new_mask = new_mask.to(torch.float32)
    tmp3= avg_pool(new_mask, old_mask.shape[-1])[0].float().cpu().repeat(80, 1)
    #print(tmp3.shape)
    tmp3=tmp3.to(torch.float32).numpy()

    ax[2].imshow(tmp1, interpolation='nearest', aspect='auto')
    ax[2].imshow(tmp3,interpolation='nearest', aspect='auto', alpha=0.2, cmap="gray")

    #print(new_mask[0,10])

    # ax[1].imshow(src_tokens[0].cpu().T.numpy(), interpolation='nearest', aspect='auto')
    # ax[1].plot(nn.functional.adaptive_avg_pool1d(new_mask.float(), old_mask.shape[-1])[0].float().cpu().numpy())


    plt.savefig("/home/gaochenghao/pic/{}_{}".format(tag, count))
    plt.clf()
    plt.cla()
    plt.close("all")



def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)
