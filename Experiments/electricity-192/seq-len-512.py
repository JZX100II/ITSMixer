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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl192_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 17709
# val 2441
# test 5069
# Epoch: 1 cost time: 57.02839684486389
# Epoch: 1, Steps: 69 | Train Loss: 0.5824657 Vali Loss: 0.3565631 Test Loss: 0.3795951
# Validation loss decreased (inf --> 0.356563).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 56.530049085617065
# Epoch: 2, Steps: 69 | Train Loss: 0.2910462 Vali Loss: 0.2115692 Test Loss: 0.2283280
# Validation loss decreased (0.356563 --> 0.211569).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 56.51620817184448
# Epoch: 3, Steps: 69 | Train Loss: 0.2273174 Vali Loss: 0.1925746 Test Loss: 0.2089700
# Validation loss decreased (0.211569 --> 0.192575).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 56.568211793899536
# Epoch: 4, Steps: 69 | Train Loss: 0.2141584 Vali Loss: 0.1853932 Test Loss: 0.2018044
# Validation loss decreased (0.192575 --> 0.185393).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 56.49377512931824
# Epoch: 5, Steps: 69 | Train Loss: 0.2082641 Vali Loss: 0.1820294 Test Loss: 0.1983552
# Validation loss decreased (0.185393 --> 0.182029).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 56.54855489730835
# Epoch: 6, Steps: 69 | Train Loss: 0.2048412 Vali Loss: 0.1790818 Test Loss: 0.1957460
# Validation loss decreased (0.182029 --> 0.179082).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 56.5184485912323
# Epoch: 7, Steps: 69 | Train Loss: 0.2026728 Vali Loss: 0.1777440 Test Loss: 0.1942627
# Validation loss decreased (0.179082 --> 0.177744).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 56.53035497665405
# Epoch: 8, Steps: 69 | Train Loss: 0.2006759 Vali Loss: 0.1762349 Test Loss: 0.1928926
# Validation loss decreased (0.177744 --> 0.176235).  Saving model ...
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 56.53845953941345
# Epoch: 9, Steps: 69 | Train Loss: 0.1991732 Vali Loss: 0.1756227 Test Loss: 0.1923425
# Validation loss decreased (0.176235 --> 0.175623).  Saving model ...
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 56.57191061973572
# Epoch: 10, Steps: 69 | Train Loss: 0.1980332 Vali Loss: 0.1744062 Test Loss: 0.1911048
# Validation loss decreased (0.175623 --> 0.174406).  Saving model ...
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 56.552632570266724
# Epoch: 11, Steps: 69 | Train Loss: 0.1969336 Vali Loss: 0.1754322 Test Loss: 0.1928849
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 56.59029746055603
# Epoch: 12, Steps: 69 | Train Loss: 0.1958505 Vali Loss: 0.1764448 Test Loss: 0.1940924
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 56.55095934867859
# Epoch: 13, Steps: 69 | Train Loss: 0.1949132 Vali Loss: 0.1775521 Test Loss: 0.1956415
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 56.54458141326904
# Epoch: 14, Steps: 69 | Train Loss: 0.1942660 Vali Loss: 0.1795810 Test Loss: 0.1973073
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 56.55045199394226
# Epoch: 15, Steps: 69 | Train Loss: 0.1933640 Vali Loss: 0.1788128 Test Loss: 0.1969205
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002824295364810001
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl192_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 5069
# mse:0.1416747272014618, mae:0.2405349761247635, rse:0.37645697593688965