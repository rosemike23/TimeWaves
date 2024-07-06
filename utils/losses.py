# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F

def autocorrelation(query_states, key_states):
    """
    Computes autocorrelation(Q,K) using `torch.fft`.
    Think about it as a replacement for the QK^T in the self-attention.
    
    Assumption: states are resized to same shape of [batch_size, time_length, embedding_dim].
    """
    query_states_fft = t.fft.rfft(query_states, dim=1)
    key_states_fft = t.fft.rfft(key_states, dim=1)
    attn_weights = query_states_fft * t.conj(key_states_fft)
    attn_weights = t.fft.irfft(attn_weights, dim=1)
    
    return attn_weights


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class TILDEQ_loss(nn.Module):
    def __init__(self):
        super(TILDEQ_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """

        #? loss a.shift
        dyy = (target-forecast)* mask
        eq = t.abs((1/forecast.shape[1]) - F.softmax(dyy, dim=1))
        loss_ashift = forecast.shape[1] * eq.sum(dim=1)


        # #? loss phase
        forecast_f = abs(t.fft.rfft(forecast, dim=1)) # 对时间维度dim=1做fft 
        target_f = abs(t.fft.rfft(target, dim=1))
        target_f[:,0] = 0; forecast_f[:,0] = 0

        target_f = t.squeeze(target_f)
        forecast_f = t.squeeze(forecast_f)
        _, top_list_target = t.topk(target_f, 5, dim=1)
        _, bottom_list_target = t.topk(-target_f, target_f.shape[1]-5, dim=1)
        loss_phase = t.zeros_like(loss_ashift)

        for i in range(top_list_target.shape[0]):
            loss_phase[i] = t.norm(target_f[i,top_list_target[i]]
                        - forecast_f[i,top_list_target[i]],p=2 ,dim=0)
            loss_phase[i] += t.norm(forecast_f[i,bottom_list_target[i]],p=2 ,dim=0)

        # #? loss auto-correlation
        # loss_amp = \
        #     t.norm(autocorrelation(target, target) - autocorrelation(forecast, target), 
        #        p=2, dim=1)

        # loss_TILDEQ = 0.99*loss_ashift + (1-0.99)*loss_phase + 0.5*loss_amp
        # loss_TILDEQ = 0.99*loss_ashift + (1-0.99)*loss_phase
        loss_TILDEQ = 0.99*loss_ashift 
        # print(loss_ashift)
        # print(loss_phase)
        # print(loss_amp)
        
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        
        return t.mean(loss_TILDEQ)/4 + 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)\
        + t.mean(t.abs(target - forecast) * masked_masep_inv)
    
