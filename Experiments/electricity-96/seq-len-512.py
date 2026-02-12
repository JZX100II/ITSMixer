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
        self.drop = nn.Dropout(0.1)

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
        self.drop = nn.Dropout(0.03)

    def forward(self, x):
        res1 = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res1

        # x = self.drop(x)

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

        # x = self.drop(x)

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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl96_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 17805
# val 2537
# test 5165
# Epoch: 1 cost time: 55.96401309967041
# Epoch: 1, Steps: 69 | Train Loss: 0.5415254 Vali Loss: 0.3334137 Test Loss: 0.3626617
# Validation loss decreased (inf --> 0.333414).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 54.43733549118042
# Epoch: 2, Steps: 69 | Train Loss: 0.2685943 Vali Loss: 0.1934619 Test Loss: 0.2141257
# Validation loss decreased (0.333414 --> 0.193462).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 54.483474016189575
# Epoch: 3, Steps: 69 | Train Loss: 0.2070876 Vali Loss: 0.1773266 Test Loss: 0.1974229
# Validation loss decreased (0.193462 --> 0.177327).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 54.38125944137573
# Epoch: 4, Steps: 69 | Train Loss: 0.1954308 Vali Loss: 0.1696772 Test Loss: 0.1904318
# Validation loss decreased (0.177327 --> 0.169677).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 54.45408344268799
# Epoch: 5, Steps: 69 | Train Loss: 0.1900985 Vali Loss: 0.1669428 Test Loss: 0.1871969
# Validation loss decreased (0.169677 --> 0.166943).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 54.49370336532593
# Epoch: 6, Steps: 69 | Train Loss: 0.1870608 Vali Loss: 0.1645610 Test Loss: 0.1852204
# Validation loss decreased (0.166943 --> 0.164561).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 54.47199368476868
# Epoch: 7, Steps: 69 | Train Loss: 0.1845951 Vali Loss: 0.1635099 Test Loss: 0.1837941
# Validation loss decreased (0.164561 --> 0.163510).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 54.47964835166931
# Epoch: 8, Steps: 69 | Train Loss: 0.1827376 Vali Loss: 0.1628964 Test Loss: 0.1838619
# Validation loss decreased (0.163510 --> 0.162896).  Saving model ...
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 54.4492666721344
# Epoch: 9, Steps: 69 | Train Loss: 0.1811942 Vali Loss: 0.1641392 Test Loss: 0.1855551
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 54.38809823989868
# Epoch: 10, Steps: 69 | Train Loss: 0.1798560 Vali Loss: 0.1633810 Test Loss: 0.1848088
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 54.32457971572876
# Epoch: 11, Steps: 69 | Train Loss: 0.1787451 Vali Loss: 0.1648353 Test Loss: 0.1862418
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 54.430887937545776
# Epoch: 12, Steps: 69 | Train Loss: 0.1776280 Vali Loss: 0.1618800 Test Loss: 0.1833840
# Validation loss decreased (0.162896 --> 0.161880).  Saving model ...
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 54.244952917099
# Epoch: 13, Steps: 69 | Train Loss: 0.1767547 Vali Loss: 0.1589445 Test Loss: 0.1795485
# Validation loss decreased (0.161880 --> 0.158945).  Saving model ...
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 54.431132793426514
# Epoch: 14, Steps: 69 | Train Loss: 0.1759395 Vali Loss: 0.1596611 Test Loss: 0.1804781
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 54.38727355003357
# Epoch: 15, Steps: 69 | Train Loss: 0.1752845 Vali Loss: 0.1611185 Test Loss: 0.1821562
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0002824295364810001
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl96_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 5165
# mse:0.13223344087600708, mae:0.22686432301998138, rse:0.36169326305389404