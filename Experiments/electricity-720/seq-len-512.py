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
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, hidden_features)
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
        self.layers = [0] * 50
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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl720_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 17181
# val 1913
# test 4541
# Epoch: 1 cost time: 34.717252254486084
# Epoch: 1, Steps: 67 | Train Loss: 0.7668821 Vali Loss: 0.5085357 Test Loss: 0.5430194
# Validation loss decreased (inf --> 0.508536).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 34.144469022750854
# Epoch: 2, Steps: 67 | Train Loss: 0.4279463 Vali Loss: 0.2834441 Test Loss: 0.3130569
# Validation loss decreased (0.508536 --> 0.283444).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 34.19595170021057
# Epoch: 3, Steps: 67 | Train Loss: 0.3112545 Vali Loss: 0.2570423 Test Loss: 0.2858743
# Validation loss decreased (0.283444 --> 0.257042).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 34.17608618736267
# Epoch: 4, Steps: 67 | Train Loss: 0.2910224 Vali Loss: 0.2443956 Test Loss: 0.2721640
# Validation loss decreased (0.257042 --> 0.244396).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 34.114091873168945
# Epoch: 5, Steps: 67 | Train Loss: 0.2822228 Vali Loss: 0.2385142 Test Loss: 0.2659962
# Validation loss decreased (0.244396 --> 0.238514).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 34.2985999584198
# Epoch: 6, Steps: 67 | Train Loss: 0.2770849 Vali Loss: 0.2344928 Test Loss: 0.2623689
# Validation loss decreased (0.238514 --> 0.234493).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 34.260931968688965
# Epoch: 7, Steps: 67 | Train Loss: 0.2737685 Vali Loss: 0.2322888 Test Loss: 0.2596482
# Validation loss decreased (0.234493 --> 0.232289).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 34.285395860672
# Epoch: 8, Steps: 67 | Train Loss: 0.2716217 Vali Loss: 0.2302626 Test Loss: 0.2579919
# Validation loss decreased (0.232289 --> 0.230263).  Saving model ...
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 34.17807602882385
# Epoch: 9, Steps: 67 | Train Loss: 0.2697222 Vali Loss: 0.2284779 Test Loss: 0.2562766
# Validation loss decreased (0.230263 --> 0.228478).  Saving model ...
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 34.261205434799194
# Epoch: 10, Steps: 67 | Train Loss: 0.2682819 Vali Loss: 0.2272578 Test Loss: 0.2552699
# Validation loss decreased (0.228478 --> 0.227258).  Saving model ...
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 34.141308307647705
# Epoch: 11, Steps: 67 | Train Loss: 0.2670505 Vali Loss: 0.2261139 Test Loss: 0.2542094
# Validation loss decreased (0.227258 --> 0.226114).  Saving model ...
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 34.113991260528564
# Epoch: 12, Steps: 67 | Train Loss: 0.2658919 Vali Loss: 0.2251530 Test Loss: 0.2532103
# Validation loss decreased (0.226114 --> 0.225153).  Saving model ...
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 34.194732666015625
# Epoch: 13, Steps: 67 | Train Loss: 0.2648801 Vali Loss: 0.2243186 Test Loss: 0.2526245
# Validation loss decreased (0.225153 --> 0.224319).  Saving model ...
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 34.0657262802124
# Epoch: 14, Steps: 67 | Train Loss: 0.2639862 Vali Loss: 0.2237083 Test Loss: 0.2519054
# Validation loss decreased (0.224319 --> 0.223708).  Saving model ...
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 34.13946080207825
# Epoch: 15, Steps: 67 | Train Loss: 0.2632533 Vali Loss: 0.2233179 Test Loss: 0.2513627
# Validation loss decreased (0.223708 --> 0.223318).  Saving model ...
# Updating learning rate to 0.0002824295364810001
# Epoch: 16 cost time: 34.150301456451416
# Epoch: 16, Steps: 67 | Train Loss: 0.2625629 Vali Loss: 0.2223011 Test Loss: 0.2506837
# Validation loss decreased (0.223318 --> 0.222301).  Saving model ...
# Updating learning rate to 0.0002541865828329001
# Epoch: 17 cost time: 34.190698862075806
# Epoch: 17, Steps: 67 | Train Loss: 0.2620165 Vali Loss: 0.2238836 Test Loss: 0.2506033
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0002287679245496101
# Epoch: 18 cost time: 34.34095048904419
# Epoch: 18, Steps: 67 | Train Loss: 0.2613569 Vali Loss: 0.2246920 Test Loss: 0.2506274
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0002058911320946491
# Epoch: 19 cost time: 34.23956656455994
# Epoch: 19, Steps: 67 | Train Loss: 0.2608926 Vali Loss: 0.2256639 Test Loss: 0.2507488
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.00018530201888518417
# Epoch: 20 cost time: 34.220168113708496
# Epoch: 20, Steps: 67 | Train Loss: 0.2605041 Vali Loss: 0.2258247 Test Loss: 0.2507688
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.00016677181699666576
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl720_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 4541
# mse:0.2048303335905075, mae:0.296536922454834, rse:0.4525635242462158