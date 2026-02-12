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
        self.drop = nn.Dropout(0.79)

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
        self.layers = [0] * 50
        self.drop = nn.Dropout(0.62)

    def forward(self, x):
        res = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res

        x = self.drop(x)
        temp = x

        for i in range(7):
            self.layers[i] = x
            x = self.batchNorm2D(x)
            x = self.MLP_time1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time2(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = x + temp

        x = self.drop(x)

        for i in range(7):
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
# >>>>>>>start training : test_ITSMixer_ETTm2_ftM_sl512_pl192_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33857
# val 11329
# test 11329
# 	iters: 100, epoch: 1 | loss: 0.4686425
# 	speed: 0.0888s/iter; left time: 108.4097s
# Epoch: 1 cost time: 10.426206827163696
# Epoch: 1, Steps: 132 | Train Loss: 0.5781307 Vali Loss: 0.2675729 Test Loss: 0.3172690
# Validation loss decreased (inf --> 0.267573).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.3496983
# 	speed: 0.1534s/iter; left time: 167.0367s
# Epoch: 2 cost time: 10.71082091331482
# Epoch: 2, Steps: 132 | Train Loss: 0.3643081 Vali Loss: 0.2128190 Test Loss: 0.2579561
# Validation loss decreased (0.267573 --> 0.212819).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.3520796
# 	speed: 0.1560s/iter; left time: 149.3367s
# Epoch: 3 cost time: 10.585060596466064
# Epoch: 3, Steps: 132 | Train Loss: 0.3134876 Vali Loss: 0.2234701 Test Loss: 0.2733779
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.3174950
# 	speed: 0.1560s/iter; left time: 128.6755s
# Epoch: 4 cost time: 10.901653528213501
# Epoch: 4, Steps: 132 | Train Loss: 0.3053842 Vali Loss: 0.2273303 Test Loss: 0.2764277
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.2981142
# 	speed: 0.1531s/iter; left time: 106.0846s
# Epoch: 5 cost time: 10.56877851486206
# Epoch: 5, Steps: 132 | Train Loss: 0.2991142 Vali Loss: 0.2294133 Test Loss: 0.2788265
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.2689980
# 	speed: 0.1557s/iter; left time: 87.3749s
# Epoch: 6 cost time: 10.688148498535156
# Epoch: 6, Steps: 132 | Train Loss: 0.2958744 Vali Loss: 0.2353020 Test Loss: 0.2839611
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.3101323
# 	speed: 0.1553s/iter; left time: 66.6024s
# Epoch: 7 cost time: 10.396339178085327
# Epoch: 7, Steps: 132 | Train Loss: 0.2905962 Vali Loss: 0.2282359 Test Loss: 0.2856426
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.2951823
# 	speed: 0.1524s/iter; left time: 45.2648s
# Epoch: 8 cost time: 10.55254602432251
# Epoch: 8, Steps: 132 | Train Loss: 0.2889207 Vali Loss: 0.2327219 Test Loss: 0.2857208
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.2821310
# 	speed: 0.1529s/iter; left time: 25.2358s
# Epoch: 9 cost time: 10.650310754776001
# Epoch: 9, Steps: 132 | Train Loss: 0.2851056 Vali Loss: 0.2294709 Test Loss: 0.2825372
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3000654
# 	speed: 0.1587s/iter; left time: 5.2383s
# Epoch: 10 cost time: 10.895132303237915
# Epoch: 10, Steps: 132 | Train Loss: 0.2840642 Vali Loss: 0.2238991 Test Loss: 0.2742335
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0004782969000000001
# >>>>>>>testing : test_ITSMixer_ETTm2_ftM_sl512_pl192_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 11329
# mse:0.22342856228351593, mae:0.29248324036598206, rse:0.3820127546787262