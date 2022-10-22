import math

import torch
import numpy as np

from math import pi
from scipy.special import logsumexp
from torch import nn

from fairseq.modules.utils import calculate_matmul, calculate_matmul_n_times


# version 0.10
# 2022/10/18
class GaussianMixture1(nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(self, n_components, n_features, covariance_type="diag", eps=1.e-6, init_params="kmeans", mu_init=None,
                 var_init=None, init_mean=None, max_iter=100, iter_per=10, gmm_prob=75):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture1, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params
        self.init_mean = init_mean

        self.max_iter = torch.nn.Parameter(torch.tensor([max_iter]), requires_grad=False)
        self.iter_per = iter_per
        self.current_iter = torch.nn.Parameter(torch.tensor([0]), requires_grad=False)

        assert self.covariance_type in ["diag", "full"]
        assert self.init_params in ["kmeans", "random"]

        self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)
        # 旧的初始化方式
        self.mu[0, 0, :] += -0.5
        self.mu[0, self.n_components-1, :] += 0.5
        if self.n_components>2:
            self.mu[0, 1, :] += 0.3

        self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.zeros(1, self.n_components, 1), requires_grad=False)
        self.pi += 1. / self.n_components

        self.device = None

        self.ratio_list = []

        self.reverse_predict = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.gmm_prob = gmm_prob
        self.count = 0

        # 初始化值
        # self.mu.data[0,0,:]+=
        #self.var.data[0, 0, :] += (torch.randn(self.n_features)/3 + 0.5)
        #self.var.data[0, 1, :] += (torch.randn(self.n_features)/3 + 0.5)

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def fit(self, x, delta=1e-3):
        # x:[seq_len x bsz , hidden_size]
        if self.current_iter >= self.max_iter:
            return
        x = self.check_size(x)

        for i in range(self.iter_per):
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            self.current_iter.data += 1
            print("GMM: iter[{}] prob:{}".format(self.current_iter.item(), self.log_likelihood))

            # 默认是80，在asr中效果不好
            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(
                    self.log_likelihood) or self.log_likelihood > -self.gmm_prob:
                print("GMM: log is inf or isnan or >{}, no update".format(self.gmm_prob))
                # if self.init_params == "kmeans":
                #     mu = self.get_kmeans_mu(x, n_centers=self.n_components)
                #     self.mu.data = mu.half()
                #     self.mu[0, 0, :] += -0.5
                #     self.mu[0, 1, :] += 0.5
                self.__update_mu(mu_old)
                self.__update_var(var_old)
                break
            else:
                j = self.log_likelihood - log_likelihood_old

                if j <= delta:
                    print("GMM: j<=delta, no update")
                    # When score decreases, revert to old parameters
                    self.__update_mu(mu_old)
                    self.__update_var(var_old)

        # if self.current_iter>=38:
        #     print(self.mu.data)
        #     print(self.var.data)

        if self.current_iter >= self.max_iter:
            # self.params_fitted = True
            # 尝试对mu增加偏置
            # print(self.mu.data)
            # self.mu.data[0,:]+=0.3
            print(self.mu.data)
            print(self.var.data)
            print("GMM: 超过最大迭代次数，GMM停止训练")

    # def predict(self,src_tokens):
    #     return self.forward(src_tokens)

    def predict(self, src_tokens):
        """
               Assigns input data to one of the mixture components by evaluating the likelihood under each.
               If probs=True returns normalized probabilities of class membership.
               args:
                   x:          torch.Tensor (n, d) or (n, 1, d)
                   probs:      bool
               returns:
                   p_k:        torch.Tensor (n, k)
                   (or)
                   y:          torch.LongTensor (n)
               """
        src_tokens=src_tokens[:,:,:self.n_features]

        src_tokens_tmp = src_tokens.reshape(-1, src_tokens.shape[-1])
        need_shape = [src_tokens.shape[0], src_tokens.shape[1]]
        if self.training:
            self.fit(src_tokens_tmp)
            # self.fit(src_tokens_tmp, n_iter=10)
        else:
            # if self.count==0:
            #     self.mu.data[0, 1, :] += 0.3
            #     print(self.var.data)
            #     self.var.data[0,0,:]+=0.2
            #     self.var.data[0,1,:] -= 0.1
            #     print(self.var.data)
            # print(self.mu.data)
            self.count += 1
        #     print("")
        # self.max_iter.data = torch.tensor([12]).to(src_tokens.device)
        # self.fit(src_tokens_tmp, n_iter=2)

        x = self.check_size(src_tokens_tmp)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        result = torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor)).to(device=x.device)

        # GMM是否需要取反
        # ratio = result.sum(-1) / result.shape[0]
        # if len(self.ratio_list)<10:
        #     self.ratio_list.append(ratio.item())
        #     if len(self.ratio_list)==10:
        #         if sum(self.ratio_list[5:]) / len(self.ratio_list[5:])<0.45:
        #             print("GMM:预测取反")
        #             self.reverse_predict.data=self.reverse_predict+1
        if self.n_components>2:
            result[result < self.n_components-1] = 0
            result[result == self.n_components-1] = 1

        result=result.bool()

        if self.reverse_predict:
            result = ~result.bool()
        else:
            result = result.bool()
        # self.pre_ratio=ratio

        # reshape
        if need_shape is not None:
            result = result.reshape(need_shape[0], need_shape[1])

        return result

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in np.arange(self.n_components)[counts > 0]:
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(
                    self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):
        """
        计算每个点属于第k个高斯分布的概率
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            precision = torch.inverse(var)
            d = x.shape[-1]

            log_2pi = d * np.log(2. * pi)

            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det

    def _print_device(self):
        print("mu:{}".format(self.mu.device))
        print("var:{}".format(self.var.device))
        print("pi:{}".format(self.pi.device))

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()

        return log_det.unsqueeze(-1)

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                            keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components,
                                                                    self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
            self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (
                1, self.n_components, self.n_features,
                self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (
                self.n_components, self.n_features, self.n_features, self.n_components, self.n_features,
                self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components,
                                                                         self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
                self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [
            (1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
            1, self.n_components, 1)

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            if i == 0 and self.init_mean is not None:
                tmp_center = self.init_mean
            else:
                tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0) * (x_max - x_min) + x_min)


"""
特征压缩
version:1.8
vad:mean,thresh
mean:特征平均值，按照百分比，去掉平均值较小的部分
thresh:特征平均值，按照固定阈值，去掉能量较小的部分
"""


class Shrink:
    def __init__(self, debug=True, mask_value=[0.1, 0.2], feature_sample=20, random_mask=False, thresh_dynamic=False,
                 method="mean"):

        self.debug = debug
        self.mask_value = mask_value
        self.feature_sample = feature_sample

        self.count = 0
        self.random_mask = random_mask
        self.thresh_dynamic = thresh_dynamic

        if "mean" in method:
            self.method = "mean"
        elif "thresh" in method:
            self.method = "thresh"
        else:
            raise RuntimeError("shrink type {} not supported".format(method))

    def set_debug(self, debug):
        self.debug = debug

    # method:vad方法
    # drop_percent:压缩比例
    # feature_sample:对前多少个特征进行采样
    def predict(self, old_src_tokens):

        # drop_percent ：去掉的比例，默认为0.1，即去除序列中平均特征值最小的10%的时间步
        if self.debug:
            print("vad方法:{}".format(self.method))
            print("输入句子长度:")
            print(old_src_tokens)

        pre_length = old_src_tokens.shape[1]

        ############################################
        ######## 此处根据不同的方法得到mask ############
        ############################################
        # mask矩阵中，True为要保留的元素

        mask = None
        if self.method == "mean":
            mask = self.vad_mean(old_src_tokens, pre_length)
        elif self.method == "thresh":
            mask = self.vad_thresh(old_src_tokens)

        if self.debug:
            print("mask:")
            print(mask)

        return mask

    def vad_mean(self, old_src_tokens, pre_length):
        src_tokens_mean = old_src_tokens[:, :, :self.feature_sample].mean(-1)

        sorted, indices = src_tokens_mean.sort(descending=False)  # 按照平均值升序排列
        need_drop_count = int(pre_length * self.mask_value[0])

        drop_value = sorted[:, need_drop_count].reshape(-1, 1)
        mask = src_tokens_mean > (drop_value - 1e-6)

        # mask矩阵中，True为要保留的元素
        return [mask]

    def vad_thresh(self, src_tokens):

        src_tokens_mean = src_tokens[:, :, :self.feature_sample].mean(-1)

        if len(self.mask_value) == 1:

            if self.thresh_dynamic:
                # 按照数据的方差，设置阈值，范围为 thresh*var
                # print("阈值:var x thresh")
                mask = src_tokens_mean > self.mask_value[0] * torch.var(src_tokens_mean)  # 设置>thresh的为高能量区域

                # print("-0.8 + 1.0 x var : 88%压缩率")
                # mask = src_tokens_mean > -0.8+1.0* torch.var(src_tokens_mean)  # 设置>thresh的为高能量区域

                # print("-0.6 + 1.0 x var : 85%压缩率")
                # mask = src_tokens_mean > -0.6 + 1.0 * torch.var(src_tokens_mean)  # 设置>thresh的为高能量区域

                # 不压缩
                # mask = src_tokens_mean > -99
                # mask = src_tokens_mean > -0.0
            else:
                mask = src_tokens_mean > self.mask_value[0]  # 设置>thresh的为高能量区域

            # print("is training:{}".format(self.is_trainning))
            if self.random_mask and self.is_trainning:
                # 训练时，随机将10%设置为低能量区域(False)
                # 20%的比例，经实验测试，不太行
                tmp = torch.rand(mask.shape[0], mask.shape[1], device=src_tokens.device)
                mask_random = tmp > 0.1
                mask = mask & mask_random

                # 随机将10%设置为高能量区域(True)
                # mask_random1 = tmp > 0.9
                # mask = mask | mask_random1

            # 异常处理，如果有句子的mask全部为false，则将其设置为全True
            error_mask = (~mask).all(-1).unsqueeze(1)
            mask = mask | error_mask

            mask = [mask]
        else:
            if self.thresh_dynamic:
                # 按照数据的方差，设置阈值
                thresh_1 = self.mask_value[0] * torch.var(src_tokens_mean)  # 设置>thresh的为高能量区域
                thresh_2 = thresh_1 + self.mask_value[1] - self.mask_value[0]
                mask_1 = src_tokens_mean > thresh_1  # 得到>0.5*var的mask
                mask_2 = src_tokens_mean > thresh_2  # 得到>0.5*var+0.3的mask

            else:
                # thresh=[0.5,0.8]
                mask_1 = src_tokens_mean > self.mask_value[0]  # 得到>0.5的mask
                mask_2 = src_tokens_mean > self.mask_value[1]  # 得到>0.8的mask

            # print("is training:{}".format(self.is_trainning))
            if self.random_mask and self.is_trainning:
                # 训练时，随机将10%设置为低能量区域(False)
                tmp = torch.rand(mask_1.shape[0], mask_1.shape[1], device=src_tokens.device)
                mask_random = tmp > 0.1
                mask_1 = mask_1 & mask_random
                mask_2 = mask_2 & mask_random

            mask = [mask_1, mask_2]

            # 异常处理，如果有句子的mask全部为false，则将其设置为全True
            error_mask = (~mask_2).all(-1).unsqueeze(1)
            mask_2 = mask_2 | error_mask
            mask_1 = mask_1 | error_mask
            mask = [mask_1, mask_2]

        # mask矩阵中，True为要保留的元素
        return mask
