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

        for i in range(9):
          self.layers[i] = x
          x = self.batchNorm2D(x)
          x = self.MLP_time1(x.permute(0, 2, 1)).permute(0, 2, 1)
          x = x + self.layers[i]

          temp = self.batchNorm2D(temp)
          temp = self.MLP_time2(temp.permute(0, 2, 1)).permute(0, 2, 1)
          temp = temp + self.layers[i]

        x = temp + x

        # x = self.drop(x)

        for i in range(9):
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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl336_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 17565
# val 2297
# test 4925
# Epoch: 1 cost time: 64.34263300895691
# Epoch: 1, Steps: 68 | Train Loss: 0.6149111 Vali Loss: 0.3814237 Test Loss: 0.4089323
# Validation loss decreased (inf --> 0.381424).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 63.64236783981323
# Epoch: 2, Steps: 68 | Train Loss: 0.3165489 Vali Loss: 0.2283573 Test Loss: 0.2512546
# Validation loss decreased (0.381424 --> 0.228357).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 63.73262906074524
# Epoch: 3, Steps: 68 | Train Loss: 0.2505617 Vali Loss: 0.2102039 Test Loss: 0.2321733
# Validation loss decreased (0.228357 --> 0.210204).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 63.73464226722717
# Epoch: 4, Steps: 68 | Train Loss: 0.2369012 Vali Loss: 0.2025177 Test Loss: 0.2243671
# Validation loss decreased (0.210204 --> 0.202518).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 63.63504600524902
# Epoch: 5, Steps: 68 | Train Loss: 0.2304325 Vali Loss: 0.1987845 Test Loss: 0.2209464
# Validation loss decreased (0.202518 --> 0.198784).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 63.64399743080139
# Epoch: 6, Steps: 68 | Train Loss: 0.2268838 Vali Loss: 0.1965622 Test Loss: 0.2186630
# Validation loss decreased (0.198784 --> 0.196562).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 63.691303968429565
# Epoch: 7, Steps: 68 | Train Loss: 0.2243932 Vali Loss: 0.1950676 Test Loss: 0.2174284
# Validation loss decreased (0.196562 --> 0.195068).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 63.72554278373718
# Epoch: 8, Steps: 68 | Train Loss: 0.2221659 Vali Loss: 0.1932689 Test Loss: 0.2158450
# Validation loss decreased (0.195068 --> 0.193269).  Saving model ...
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 63.67278504371643
# Epoch: 9, Steps: 68 | Train Loss: 0.2203823 Vali Loss: 0.1921844 Test Loss: 0.2148851
# Validation loss decreased (0.193269 --> 0.192184).  Saving model ...
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 63.7636079788208
# Epoch: 10, Steps: 68 | Train Loss: 0.2189625 Vali Loss: 0.1919803 Test Loss: 0.2143255
# Validation loss decreased (0.192184 --> 0.191980).  Saving model ...
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 63.731818437576294
# Epoch: 11, Steps: 68 | Train Loss: 0.2176751 Vali Loss: 0.1937466 Test Loss: 0.2159602
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 63.75243544578552
# Epoch: 12, Steps: 68 | Train Loss: 0.2166101 Vali Loss: 0.1984440 Test Loss: 0.2214211
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 63.68858051300049
# Epoch: 13, Steps: 68 | Train Loss: 0.2156675 Vali Loss: 0.2033609 Test Loss: 0.2262129
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 63.703715801239014
# Epoch: 14, Steps: 68 | Train Loss: 0.2147591 Vali Loss: 0.2045661 Test Loss: 0.2285098
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 63.681560039520264
# Epoch: 15, Steps: 68 | Train Loss: 0.2137764 Vali Loss: 0.2018519 Test Loss: 0.2258892
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002824295364810001
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl336_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 4925
# mse:0.16600269079208374, mae:0.26264750957489014, rse:0.4061364531517029