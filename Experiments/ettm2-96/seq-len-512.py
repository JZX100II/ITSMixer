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
        self.layers = [0] * 10
        self.drop = nn.Dropout(0.35)

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
# >>>>>>>start training : test_ITSMixer_ETTm2_ftM_sl512_pl96_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33953
# val 11425
# test 11425
# 	iters: 100, epoch: 1 | loss: 0.3699841
# 	speed: 0.0871s/iter; left time: 106.3593s
# Epoch: 1 cost time: 10.112808227539062
# Epoch: 1, Steps: 132 | Train Loss: 0.4592140 Vali Loss: 0.2302544 Test Loss: 0.2752575
# Validation loss decreased (inf --> 0.230254).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.2549958
# 	speed: 0.1453s/iter; left time: 158.2586s
# Epoch: 2 cost time: 10.123372316360474
# Epoch: 2, Steps: 132 | Train Loss: 0.2865231 Vali Loss: 0.1752014 Test Loss: 0.2114403
# Validation loss decreased (0.230254 --> 0.175201).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.2214524
# 	speed: 0.1493s/iter; left time: 142.9131s
# Epoch: 3 cost time: 10.0207839012146
# Epoch: 3, Steps: 132 | Train Loss: 0.2477920 Vali Loss: 0.1770884 Test Loss: 0.2142194
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.2844794
# 	speed: 0.1510s/iter; left time: 124.5932s
# Epoch: 4 cost time: 10.275137424468994
# Epoch: 4, Steps: 132 | Train Loss: 0.2382246 Vali Loss: 0.1774600 Test Loss: 0.2149813
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.2219879
# 	speed: 0.1550s/iter; left time: 107.3991s
# Epoch: 5 cost time: 10.474791526794434
# Epoch: 5, Steps: 132 | Train Loss: 0.2315964 Vali Loss: 0.1793870 Test Loss: 0.2214349
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.2502114
# 	speed: 0.1520s/iter; left time: 85.2998s
# Epoch: 6 cost time: 10.752806901931763
# Epoch: 6, Steps: 132 | Train Loss: 0.2271905 Vali Loss: 0.1779073 Test Loss: 0.2175679
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.2149310
# 	speed: 0.1585s/iter; left time: 68.0033s
# Epoch: 7 cost time: 10.562959671020508
# Epoch: 7, Steps: 132 | Train Loss: 0.2226947 Vali Loss: 0.1777722 Test Loss: 0.2171982
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.2401319
# 	speed: 0.1521s/iter; left time: 45.1664s
# Epoch: 8 cost time: 10.416928052902222
# Epoch: 8, Steps: 132 | Train Loss: 0.2194818 Vali Loss: 0.1777147 Test Loss: 0.2203772
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.2429949
# 	speed: 0.1490s/iter; left time: 24.5867s
# Epoch: 9 cost time: 10.42241621017456
# Epoch: 9, Steps: 132 | Train Loss: 0.2171285 Vali Loss: 0.1773960 Test Loss: 0.2180853
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.2243960
# 	speed: 0.1473s/iter; left time: 4.8619s
# Epoch: 10 cost time: 10.369364500045776
# Epoch: 10, Steps: 132 | Train Loss: 0.2139738 Vali Loss: 0.1797287 Test Loss: 0.2234058
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0004782969000000001
# >>>>>>>testing : test_ITSMixer_ETTm2_ftM_sl512_pl96_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 11425
# mse:0.1677454113960266, mae:0.255135178565979, rse:0.3307778239250183