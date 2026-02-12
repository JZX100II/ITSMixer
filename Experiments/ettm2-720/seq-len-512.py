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
        self.drop = nn.Dropout(0.32)

    def forward(self, x):
        res = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res

        x = self.drop(x)
        temp = x

        for i in range(10):
            self.layers[i] = x
            x = self.batchNorm2D(x)
            x = self.MLP_time1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time2(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = temp + x

        x = self.drop(x)

        for i in range(10):
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
# >>>>>>>start training : test_ITSMixer_ETTm2_ftM_sl512_pl720_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33329
# val 10801
# test 10801
# 	iters: 100, epoch: 1 | loss: 0.5875939
# 	speed: 0.1231s/iter; left time: 147.8756s
# Epoch: 1 cost time: 14.528968334197998
# Epoch: 1, Steps: 130 | Train Loss: 0.6307044 Vali Loss: 0.3415101 Test Loss: 0.4124428
# Validation loss decreased (inf --> 0.341510).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.5563871
# 	speed: 0.2183s/iter; left time: 233.7582s
# Epoch: 2 cost time: 14.576656103134155
# Epoch: 2, Steps: 130 | Train Loss: 0.4887740 Vali Loss: 0.3069074 Test Loss: 0.3657089
# Validation loss decreased (0.341510 --> 0.306907).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.4323950
# 	speed: 0.2266s/iter; left time: 213.2551s
# Epoch: 3 cost time: 14.962035894393921
# Epoch: 3, Steps: 130 | Train Loss: 0.4482357 Vali Loss: 0.3200986 Test Loss: 0.3850600
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.4314422
# 	speed: 0.2244s/iter; left time: 181.9674s
# Epoch: 4 cost time: 15.321338653564453
# Epoch: 4, Steps: 130 | Train Loss: 0.4282157 Vali Loss: 0.3447733 Test Loss: 0.4239423
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.3942020
# 	speed: 0.2174s/iter; left time: 148.0518s
# Epoch: 5 cost time: 14.64309549331665
# Epoch: 5, Steps: 130 | Train Loss: 0.4173408 Vali Loss: 0.3486738 Test Loss: 0.4366471
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.4991019
# 	speed: 0.2240s/iter; left time: 123.4434s
# Epoch: 6 cost time: 14.536550521850586
# Epoch: 6, Steps: 130 | Train Loss: 0.4088868 Vali Loss: 0.3651754 Test Loss: 0.4457847
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.4057471
# 	speed: 0.2255s/iter; left time: 94.9169s
# Epoch: 7 cost time: 14.685885906219482
# Epoch: 7, Steps: 130 | Train Loss: 0.4011609 Vali Loss: 0.3596444 Test Loss: 0.4428192
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.3550900
# 	speed: 0.2211s/iter; left time: 64.3358s
# Epoch: 8 cost time: 15.113051176071167
# Epoch: 8, Steps: 130 | Train Loss: 0.3952249 Vali Loss: 0.3735235 Test Loss: 0.4553507
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.4163374
# 	speed: 0.2279s/iter; left time: 36.6858s
# Epoch: 9 cost time: 14.971724510192871
# Epoch: 9, Steps: 130 | Train Loss: 0.3902058 Vali Loss: 0.3830965 Test Loss: 0.4656018
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3429273
# 	speed: 0.2193s/iter; left time: 6.7995s
# Epoch: 10 cost time: 14.543360948562622
# Epoch: 10, Steps: 130 | Train Loss: 0.3865842 Vali Loss: 0.3817174 Test Loss: 0.4525377
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0004782969000000001
# >>>>>>>testing : test_ITSMixer_ETTm2_ftM_sl512_pl720_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 10801
# mse:0.35232189297676086, mae:0.379096657037735, rse:0.4765624403953552