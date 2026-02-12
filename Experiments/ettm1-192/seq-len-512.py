import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        if hasattr(self, 'affine_weight') and hasattr(self, 'affine_bias'):
            self.affine_weight.data.fill_(1.0)
            self.affine_bias.data.fill_(0.0)
        else:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class Mlp_time(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_time, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.8)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)
        
        return x

class Mixer_Layer(nn.Module):
    def __init__(self, time_dim, feat_dim):
        super(Mixer_Layer, self).__init__()
        self.batchNorm2D = nn.BatchNorm1d(time_dim)
        self.MLP_time = Mlp_time(time_dim, time_dim)
        self.MLP_time1 = Mlp_time(time_dim, time_dim)
        self.MLP_time2 = Mlp_time(time_dim, time_dim)
        self.MLP_time3 = Mlp_time(time_dim, time_dim)
        self.layers = [0] * 10
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        res = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res

        x = self.drop(x)
        temp = x

        for i in range(8):
            self.layers[i] = x
            x = self.batchNorm2D(x)
            x = self.MLP_time1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time2(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = temp + x

        x = self.drop(x)

        for i in range(8):
            x = self.batchNorm2D(x)
            x = self.MLP_time3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

        return x

class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = seq_len = configs.seq_len
        self.pred_len = pred_len = configs.pred_len
        self.enc_in = enc_in = configs.enc_in
        self.layer_num = layer_num = 10
        self.mix_layer = Mixer_Layer(seq_len, enc_in)
        self.temp_proj = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = self.mix_layer(x)
        x = self.temp_proj(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x

class Model(nn.Module):
    """ITSMixer: Iterative Time-Mixing MLP model for long-term forecasting."""
    def __init__(self, configs):
        super(Model, self).__init__()
        self.rev = RevIN(configs.enc_in)
        self.backbone = Backbone(configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):
        z = self.rev(x, 'norm')
        z = self.backbone(z)
        z = self.rev(z, 'denorm')
        return z

# Use GPU: cuda:0
# >>>>>>>start training : test_ITSMixer_ETTm1_ftM_sl512_pl192_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33857
# val 11329
# test 11329
# 	iters: 100, epoch: 1 | loss: 0.5576147
# 	speed: 0.1013s/iter; left time: 230.6918s
# Epoch: 1 cost time: 12.029075622558594
# Epoch: 1, Steps: 132 | Train Loss: 0.6223262 Vali Loss: 0.6360633 Test Loss: 0.4660735
# Validation loss decreased (inf --> 0.636063).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.3751465
# 	speed: 0.1743s/iter; left time: 373.9667s
# Epoch: 2 cost time: 12.204976558685303
# Epoch: 2, Steps: 132 | Train Loss: 0.3902483 Vali Loss: 0.5021083 Test Loss: 0.3540923
# Validation loss decreased (0.636063 --> 0.502108).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.3371951
# 	speed: 0.1820s/iter; left time: 366.4317s
# Epoch: 3 cost time: 12.650161027908325
# Epoch: 3, Steps: 132 | Train Loss: 0.3462736 Vali Loss: 0.4923932 Test Loss: 0.3492267
# Validation loss decreased (0.502108 --> 0.492393).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.3616482
# 	speed: 0.1729s/iter; left time: 325.2961s
# Epoch: 4 cost time: 12.13565993309021
# Epoch: 4, Steps: 132 | Train Loss: 0.3367434 Vali Loss: 0.4839911 Test Loss: 0.3482251
# Validation loss decreased (0.492393 --> 0.483991).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.3318289
# 	speed: 0.1768s/iter; left time: 309.2706s
# Epoch: 5 cost time: 11.951176166534424
# Epoch: 5, Steps: 132 | Train Loss: 0.3315529 Vali Loss: 0.4809023 Test Loss: 0.3507751
# Validation loss decreased (0.483991 --> 0.480902).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.3254075
# 	speed: 0.1710s/iter; left time: 276.4339s
# Epoch: 6 cost time: 11.909188032150269
# Epoch: 6, Steps: 132 | Train Loss: 0.3277848 Vali Loss: 0.4758757 Test Loss: 0.3461040
# Validation loss decreased (0.480902 --> 0.475876).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.3121215
# 	speed: 0.1742s/iter; left time: 258.7430s
# Epoch: 7 cost time: 12.12190580368042
# Epoch: 7, Steps: 132 | Train Loss: 0.3240006 Vali Loss: 0.4781426 Test Loss: 0.3454021
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.3039607
# 	speed: 0.1762s/iter; left time: 238.4031s
# Epoch: 8 cost time: 12.110332489013672
# Epoch: 8, Steps: 132 | Train Loss: 0.3205145 Vali Loss: 0.4806799 Test Loss: 0.3468449
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.3221921
# 	speed: 0.1741s/iter; left time: 212.5367s
# Epoch: 9 cost time: 12.261587619781494
# Epoch: 9, Steps: 132 | Train Loss: 0.3170350 Vali Loss: 0.4756204 Test Loss: 0.3456180
# Validation loss decreased (0.475876 --> 0.475620).  Saving model ...
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3119260
# 	speed: 0.1749s/iter; left time: 190.4885s
# Epoch: 10 cost time: 11.992916107177734
# Epoch: 10, Steps: 132 | Train Loss: 0.3141093 Vali Loss: 0.4737778 Test Loss: 0.3426242
# Validation loss decreased (0.475620 --> 0.473778).  Saving model ...
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.3063444
# 	speed: 0.1715s/iter; left time: 164.1492s
# Epoch: 11 cost time: 12.00361680984497
# Epoch: 11, Steps: 132 | Train Loss: 0.3115888 Vali Loss: 0.4824706 Test Loss: 0.3440527
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.3066452
# 	speed: 0.1747s/iter; left time: 144.1602s
# Epoch: 12 cost time: 11.90984058380127
# Epoch: 12, Steps: 132 | Train Loss: 0.3089462 Vali Loss: 0.4842452 Test Loss: 0.3463703
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.3088315
# 	speed: 0.1707s/iter; left time: 118.3259s
# Epoch: 13 cost time: 12.056697845458984
# Epoch: 13, Steps: 132 | Train Loss: 0.3069469 Vali Loss: 0.4829995 Test Loss: 0.3437420
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.2928060
# 	speed: 0.1752s/iter; left time: 98.3071s
# Epoch: 14 cost time: 12.419352293014526
# Epoch: 14, Steps: 132 | Train Loss: 0.3050420 Vali Loss: 0.4813150 Test Loss: 0.3437491
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.3077760
# 	speed: 0.1786s/iter; left time: 76.6327s
# Epoch: 15 cost time: 11.981736898422241
# Epoch: 15, Steps: 132 | Train Loss: 0.3032540 Vali Loss: 0.4826716 Test Loss: 0.3447019
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002824295364810001
# 	iters: 100, epoch: 16 | loss: 0.3054961
# 	speed: 0.1709s/iter; left time: 50.7500s
# Epoch: 16 cost time: 12.093805074691772
# Epoch: 16, Steps: 132 | Train Loss: 0.3018971 Vali Loss: 0.4815594 Test Loss: 0.3439182
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0002541865828329001
# 	iters: 100, epoch: 17 | loss: 0.3094467
# 	speed: 0.1833s/iter; left time: 30.2394s
# Epoch: 17 cost time: 12.004707098007202
# Epoch: 17, Steps: 132 | Train Loss: 0.3004630 Vali Loss: 0.4834114 Test Loss: 0.3453327
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.0002287679245496101
# 	iters: 100, epoch: 18 | loss: 0.2953377
# 	speed: 0.1722s/iter; left time: 5.6831s
# Epoch: 18 cost time: 12.05475378036499
# Epoch: 18, Steps: 132 | Train Loss: 0.2989880 Vali Loss: 0.4838673 Test Loss: 0.3446853
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0002058911320946491
# >>>>>>>testing : test_ITSMixer_ETTm1_ftM_sl512_pl192_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 11329
# mse:0.3222114145755768, mae:0.3630368411540985, rse:0.5399019718170166