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
        self.drop = nn.Dropout(0.45)

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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl96_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 36280
# val 5175
# test 10444
# 	iters: 100, epoch: 1 | loss: 0.7814053
# 	speed: 0.2329s/iter; left time: 469.4878s
# Epoch: 1 cost time: 31.479896306991577
# Epoch: 1, Steps: 141 | Train Loss: 0.7956384 Vali Loss: 0.4727370 Test Loss: 0.2618145
# Validation loss decreased (inf --> 0.472737).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.3895856
# 	speed: 0.3956s/iter; left time: 741.7392s
# Epoch: 2 cost time: 32.66389751434326
# Epoch: 2, Steps: 141 | Train Loss: 0.4585851 Vali Loss: 0.3586617 Test Loss: 0.1977445
# Validation loss decreased (0.472737 --> 0.358662).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.4184307
# 	speed: 0.4079s/iter; left time: 707.3047s
# Epoch: 3 cost time: 32.993976354599
# Epoch: 3, Steps: 141 | Train Loss: 0.3893709 Vali Loss: 0.3491990 Test Loss: 0.1897170
# Validation loss decreased (0.358662 --> 0.349199).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.3483543
# 	speed: 0.4072s/iter; left time: 648.7198s
# Epoch: 4 cost time: 32.66314220428467
# Epoch: 4, Steps: 141 | Train Loss: 0.3779154 Vali Loss: 0.3396860 Test Loss: 0.1806303
# Validation loss decreased (0.349199 --> 0.339686).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.3422148
# 	speed: 0.4076s/iter; left time: 591.8121s
# Epoch: 5 cost time: 32.79383444786072
# Epoch: 5, Steps: 141 | Train Loss: 0.3691779 Vali Loss: 0.3353602 Test Loss: 0.1762509
# Validation loss decreased (0.339686 --> 0.335360).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.3767310
# 	speed: 0.4060s/iter; left time: 532.2784s
# Epoch: 6 cost time: 33.17886686325073
# Epoch: 6, Steps: 141 | Train Loss: 0.3632258 Vali Loss: 0.3353724 Test Loss: 0.1751920
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.3954355
# 	speed: 0.4051s/iter; left time: 473.9668s
# Epoch: 7 cost time: 32.76915454864502
# Epoch: 7, Steps: 141 | Train Loss: 0.3598327 Vali Loss: 0.3300638 Test Loss: 0.1743192
# Validation loss decreased (0.335360 --> 0.330064).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.3577974
# 	speed: 0.4092s/iter; left time: 421.1131s
# Epoch: 8 cost time: 32.846866846084595
# Epoch: 8, Steps: 141 | Train Loss: 0.3556755 Vali Loss: 0.3274865 Test Loss: 0.1724205
# Validation loss decreased (0.330064 --> 0.327486).  Saving model ...
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.3102198
# 	speed: 0.4038s/iter; left time: 358.5360s
# Epoch: 9 cost time: 32.727229595184326
# Epoch: 9, Steps: 141 | Train Loss: 0.3531833 Vali Loss: 0.3267133 Test Loss: 0.1717071
# Validation loss decreased (0.327486 --> 0.326713).  Saving model ...
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3638899
# 	speed: 0.4092s/iter; left time: 305.6395s
# Epoch: 10 cost time: 33.28298544883728
# Epoch: 10, Steps: 141 | Train Loss: 0.3497978 Vali Loss: 0.3255587 Test Loss: 0.1712749
# Validation loss decreased (0.326713 --> 0.325559).  Saving model ...
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.2766219
# 	speed: 0.4072s/iter; left time: 246.7664s
# Epoch: 11 cost time: 32.784950971603394
# Epoch: 11, Steps: 141 | Train Loss: 0.3490886 Vali Loss: 0.3247317 Test Loss: 0.1713396
# Validation loss decreased (0.325559 --> 0.324732).  Saving model ...
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.3339556
# 	speed: 0.4060s/iter; left time: 188.8057s
# Epoch: 12 cost time: 32.746495962142944
# Epoch: 12, Steps: 141 | Train Loss: 0.3475801 Vali Loss: 0.3257451 Test Loss: 0.1709210
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.3292681
# 	speed: 0.4020s/iter; left time: 130.2640s
# Epoch: 13 cost time: 32.74029016494751
# Epoch: 13, Steps: 141 | Train Loss: 0.3465272 Vali Loss: 0.3269257 Test Loss: 0.1710470
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.3706428
# 	speed: 0.4110s/iter; left time: 75.2109s
# Epoch: 14 cost time: 33.216418743133545
# Epoch: 14, Steps: 141 | Train Loss: 0.3445609 Vali Loss: 0.3245350 Test Loss: 0.1700720
# Validation loss decreased (0.324732 --> 0.324535).  Saving model ...
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.3574393
# 	speed: 0.4124s/iter; left time: 17.3217s
# Epoch: 15 cost time: 32.84780478477478
# Epoch: 15, Steps: 141 | Train Loss: 0.3428775 Vali Loss: 0.3253341 Test Loss: 0.1718943
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0002824295364810001
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl96_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 10444
# mse:0.1472025364637375, mae:0.19294147193431854, rse:0.5050594806671143