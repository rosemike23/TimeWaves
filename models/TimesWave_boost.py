import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1, Inception_Block_V2
# import ptwt   # GPU acc wavelet plugin
import pywt
import numpy as np
import matplotlib.pyplot as plt



def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1) # fft towards dim=1
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    # print("Using TimeWaves")
    return period, top_list, abs(xf).mean(-1)[:, top_list]

def WT_for_Period(x, top_list, k=2):
    # [B, T, C]
    # B:batch-size T: serie length C: channel number

    B,T,C = x.shape
    cwtmatr_list = np.zeros([B,C,T-1,T])

    y = x.mean(2).mean(0).detach().cpu().numpy()
    wavename = 'cgau8' # Mother wavelet
    totalscal = int(T) 
    fc = pywt.central_frequency(wavename)  # center frequency
    cparam = 2 * fc * totalscal  
    scales = cparam / np.arange(totalscal, 1, -1) 
    # [cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0/T)
    [cwtmatr, frequencies] = pywt.cwtline(y, scales, wavename, 1.0/T, top_index=top_list)
    cw_map = abs(cwtmatr)
    # WT = cw_map.sum(0)
    # _, top_list_WT = torch.topk(torch.tensor(WT).cuda(), k)
    
    #! visualization for freq-wise intervals
    # plt.figure(figsize=(9, 9))
    # t = np.arange(0, 143, 1)
    # t1 = np.arange(0, T, 1)
    # ax = plt.gca()
    # # plt.contourf(t1, frequencies, abs(cw_map).detach().cpu().numpy())
    # contour = ax.contourf(t1, frequencies, abs(cw_map).detach().cpu().numpy())
    # plt.ylabel(u"freq(Hz)")
    # plt.xlabel(u"time(s)")
    # # plt.ylim(0,144)
    # plt.savefig('wavelet_average_channel_batch11.png')
    # # print("save pics")
    
    idx_array = np.zeros([top_list.shape[0],2])
    for i in range(top_list.shape[0]):
        # temp = T - top_list[i]*2 if T - top_list[i]*2 >= 0 else 0
        # temp = np.where(abs(frequencies-top_list[i])< 1e-3)
        # print(temp)
        cw_map_line = cw_map[i]

        threshold = 0.618
        if np.max(cw_map_line) >= 0.1:
            cw_map_line /= np.max(cw_map_line)
            threshold = 0.9
        # threshold = 0.618 # 0.75(loss=19.58), 0.65(loss=18.73),0.618(loss=18.70)
        if (cw_map_line>threshold).any():
            idx_list = np.where(cw_map_line>threshold)
            # idx_list = idx_list[1]
            idx_min, idx_max = np.min(idx_list), np.max(idx_list)
            idx_min = int(idx_min - T//20)
            idx_min = idx_min if idx_min>=0 else 0
            idx_max = int(idx_max + T//20)
            idx_max = idx_max if idx_max<=T-1 else T-1
            if idx_max-idx_min <= T//10:
                idx_min = int(idx_min - T//20)
                idx_min = idx_min if idx_min>=0 else 0
                idx_max = int(idx_max + T//20)
                idx_max = idx_max if idx_max<=T-1 else T-1
        else:
            idx_min = 0; idx_max = T-1
            
        idx_array[i,0], idx_array[i,1] = idx_min, idx_max
        
        # cw_map[temp,idx_min:idx_max] = np.max(cw_map)
    
    # plt.figure(figsize=(14, 12))
    # t1 = np.arange(0, T, 1)
    # t2 = np.arange(0, T, 1)
    # ax = plt.gca()
    # # plt.contourf(t1, frequencies, abs(cw_map).detach().cpu().numpy())
    # contour = ax.contourf(t1, frequencies, abs(cw_map))
    # cbar = plt.colorbar(contour)
    # plt.ylabel(u"freq(Hz)")
    # plt.xlabel(u"time(s)")
    # # plt.ylim(0,144)
    # plt.savefig('contour_with_lines_hours.png')
    # # print("save pics")
        
    return idx_array



class TwinBlock(nn.Module):
    def __init__(self, configs):
        super(TwinBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.w = nn.Parameter(0.5*torch.ones(2))
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        self.conv_wt = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            # Inception_Block_V1(configs.d_ff, configs.d_model,
            #                    num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, top_list, period_weight = FFT_for_Period(x, self.k)

        idx_array = WT_for_Period(x, top_list, self.k)
        # print(idx_array)
        res = []
        res_wt = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x


            x_split = x[:,int(idx_array[i,0]):int(idx_array[i,1])+1,:]
            x_split_length = x_split.shape[1]
            if (x_split.shape[1]) % period != 0:
                length_split = ((x_split.shape[1] // period) + 1) * period
                padding = torch.zeros([x_split.shape[0], (length_split - x_split.shape[1]), x_split.shape[2]]).to(x_split.device)
                x_split = torch.cat([x_split, padding], dim=1)
            else:
                length_split = x_split.shape[1]
                x_split = x_split

            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            
            x_split = x_split.reshape(B, length_split // period, period,
                            N).permute(0, 3, 1, 2).contiguous()
            
            # 2D conv: from 1d Variation to 2d Variation
            # res_out = torch.zeros_like(out); res_out = out
            out = self.conv(out)
            # out = self.conv1(out+res_out)
            x_split = self.conv_wt(x_split)
            
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            x_split = x_split.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])


            # Zero Padding
            x_split_whole =  torch.zeros([B,T,N]).cuda()
            x_split_whole[:,int(idx_array[i,0]):int(idx_array[i,1])+1,:] = x_split[:, :x_split_length, :].cuda()
            # x_split_whole = x_split[:, :x_split_length, :].cuda().resize(B,T,N)

            res_wt.append(x_split_whole)
        res = torch.stack(res, dim=-1)
        res_wt = torch.stack(res_wt, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res_wt = torch.sum(res_wt * period_weight, -1)
        # residual connection
        res = self.w[0]*res + self.w[1]*res_wt + x
        return res


class Model(nn.Module):
    """
    Reference Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TwinBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimeWaves
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimeWaves
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimeWaves
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimeWaves
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
