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
# >>>>>>>start training : test_ITSMixer_ETTm1_ftM_sl512_pl336_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33713
# val 11185
# test 11185
# 	iters: 100, epoch: 1 | loss: 0.5765063
# 	speed: 0.1018s/iter; left time: 323.3392s
# Epoch: 1 cost time: 11.825788974761963
# Epoch: 1, Steps: 131 | Train Loss: 0.6359916 Vali Loss: 0.7246534 Test Loss: 0.4836610
# Validation loss decreased (inf --> 0.724653).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.4134046
# 	speed: 0.1709s/iter; left time: 520.5172s
# Epoch: 2 cost time: 11.392600536346436
# Epoch: 2, Steps: 131 | Train Loss: 0.4219824 Vali Loss: 0.5973752 Test Loss: 0.3771499
# Validation loss decreased (0.724653 --> 0.597375).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.3751832
# 	speed: 0.1667s/iter; left time: 485.8043s
# Epoch: 3 cost time: 11.47490668296814
# Epoch: 3, Steps: 131 | Train Loss: 0.3789810 Vali Loss: 0.5946584 Test Loss: 0.3728581
# Validation loss decreased (0.597375 --> 0.594658).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.3747256
# 	speed: 0.1716s/iter; left time: 477.5061s
# Epoch: 4 cost time: 11.95052695274353
# Epoch: 4, Steps: 131 | Train Loss: 0.3707504 Vali Loss: 0.5972779 Test Loss: 0.3711863
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.3921304
# 	speed: 0.1750s/iter; left time: 464.0811s
# Epoch: 5 cost time: 11.8659508228302
# Epoch: 5, Steps: 131 | Train Loss: 0.3659490 Vali Loss: 0.5913637 Test Loss: 0.3709186
# Validation loss decreased (0.594658 --> 0.591364).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.3865895
# 	speed: 0.1709s/iter; left time: 430.7699s
# Epoch: 6 cost time: 11.842252731323242
# Epoch: 6, Steps: 131 | Train Loss: 0.3620570 Vali Loss: 0.5991243 Test Loss: 0.3702798
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.3683790
# 	speed: 0.1741s/iter; left time: 416.1582s
# Epoch: 7 cost time: 11.533107995986938
# Epoch: 7, Steps: 131 | Train Loss: 0.3585604 Vali Loss: 0.5986826 Test Loss: 0.3726994
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.3653580
# 	speed: 0.1700s/iter; left time: 383.9894s
# Epoch: 8 cost time: 11.714518785476685
# Epoch: 8, Steps: 131 | Train Loss: 0.3549634 Vali Loss: 0.5964623 Test Loss: 0.3687992
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.3585887
# 	speed: 0.1724s/iter; left time: 366.8782s
# Epoch: 9 cost time: 11.9900963306427
# Epoch: 9, Steps: 131 | Train Loss: 0.3514951 Vali Loss: 0.6064286 Test Loss: 0.3736192
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3383372
# 	speed: 0.1766s/iter; left time: 352.7636s
# Epoch: 10 cost time: 11.701953887939453
# Epoch: 10, Steps: 131 | Train Loss: 0.3488733 Vali Loss: 0.6096787 Test Loss: 0.3748344
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.3511466
# 	speed: 0.1691s/iter; left time: 315.4549s
# Epoch: 11 cost time: 11.735010862350464
# Epoch: 11, Steps: 131 | Train Loss: 0.3460990 Vali Loss: 0.6066303 Test Loss: 0.3724632
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.3369727
# 	speed: 0.1733s/iter; left time: 300.6980s
# Epoch: 12 cost time: 11.856730937957764
# Epoch: 12, Steps: 131 | Train Loss: 0.3438305 Vali Loss: 0.6036293 Test Loss: 0.3740961
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.3411392
# 	speed: 0.1733s/iter; left time: 278.0430s
# Epoch: 13 cost time: 11.705276727676392
# Epoch: 13, Steps: 131 | Train Loss: 0.3415565 Vali Loss: 0.6132297 Test Loss: 0.3748353
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.3351580
# 	speed: 0.1684s/iter; left time: 248.1046s
# Epoch: 14 cost time: 11.706274032592773
# Epoch: 14, Steps: 131 | Train Loss: 0.3392917 Vali Loss: 0.6059361 Test Loss: 0.3752784
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.3505250
# 	speed: 0.1741s/iter; left time: 233.6082s
# Epoch: 15 cost time: 11.689527034759521
# Epoch: 15, Steps: 131 | Train Loss: 0.3379078 Vali Loss: 0.6033028 Test Loss: 0.3750382
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.0002824295364810001
# 	iters: 100, epoch: 16 | loss: 0.3279901
# 	speed: 0.1688s/iter; left time: 204.3805s
# Epoch: 16 cost time: 11.64768648147583
# Epoch: 16, Steps: 131 | Train Loss: 0.3360348 Vali Loss: 0.6036300 Test Loss: 0.3742372
# EarlyStopping counter: 11 out of 100
# Updating learning rate to 0.0002541865828329001
# 	iters: 100, epoch: 17 | loss: 0.3450695
# 	speed: 0.1738s/iter; left time: 187.7252s
# Epoch: 17 cost time: 12.07536244392395
# Epoch: 17, Steps: 131 | Train Loss: 0.3347976 Vali Loss: 0.6049137 Test Loss: 0.3749160
# EarlyStopping counter: 12 out of 100
# Updating learning rate to 0.0002287679245496101
# 	iters: 100, epoch: 18 | loss: 0.3235248
# 	speed: 0.1755s/iter; left time: 166.5321s
# Epoch: 18 cost time: 11.684458017349243
# Epoch: 18, Steps: 131 | Train Loss: 0.3337307 Vali Loss: 0.6050494 Test Loss: 0.3792315
# EarlyStopping counter: 13 out of 100
# Updating learning rate to 0.0002058911320946491
# 	iters: 100, epoch: 19 | loss: 0.3226205
# 	speed: 0.1688s/iter; left time: 138.1121s
# Epoch: 19 cost time: 11.711461305618286
# Epoch: 19, Steps: 131 | Train Loss: 0.3323463 Vali Loss: 0.6034828 Test Loss: 0.3787721
# EarlyStopping counter: 14 out of 100
# Updating learning rate to 0.00018530201888518417
# 	iters: 100, epoch: 20 | loss: 0.3282505
# 	speed: 0.1731s/iter; left time: 118.8993s
# Epoch: 20 cost time: 11.637193202972412
# Epoch: 20, Steps: 131 | Train Loss: 0.3315457 Vali Loss: 0.6003622 Test Loss: 0.3760657
# EarlyStopping counter: 15 out of 100
# Updating learning rate to 0.00016677181699666576
# 	iters: 100, epoch: 21 | loss: 0.3194512
# 	speed: 0.1819s/iter; left time: 101.1605s
# Epoch: 21 cost time: 11.661710500717163
# Epoch: 21, Steps: 131 | Train Loss: 0.3304164 Vali Loss: 0.6015999 Test Loss: 0.3769314
# EarlyStopping counter: 16 out of 100
# Updating learning rate to 0.00015009463529699917
# 	iters: 100, epoch: 22 | loss: 0.3236804
# 	speed: 0.1740s/iter; left time: 73.9582s
# Epoch: 22 cost time: 12.161787033081055
# Epoch: 22, Steps: 131 | Train Loss: 0.3297799 Vali Loss: 0.6025234 Test Loss: 0.3769410
# EarlyStopping counter: 17 out of 100
# Updating learning rate to 0.0001350851717672993
# 	iters: 100, epoch: 23 | loss: 0.3286722
# 	speed: 0.1755s/iter; left time: 51.6085s
# Epoch: 23 cost time: 11.719347953796387
# Epoch: 23, Steps: 131 | Train Loss: 0.3290732 Vali Loss: 0.6032100 Test Loss: 0.3774042
# EarlyStopping counter: 18 out of 100
# Updating learning rate to 0.00012157665459056935
# 	iters: 100, epoch: 24 | loss: 0.3419732
# 	speed: 0.1707s/iter; left time: 27.8295s
# Epoch: 24 cost time: 11.751770973205566
# Epoch: 24, Steps: 131 | Train Loss: 0.3283987 Vali Loss: 0.6020838 Test Loss: 0.3763414
# EarlyStopping counter: 19 out of 100
# Updating learning rate to 0.00010941898913151242
# 	iters: 100, epoch: 25 | loss: 0.3340268
# 	speed: 0.1737s/iter; left time: 5.5573s
# Epoch: 25 cost time: 11.6855788230896
# Epoch: 25, Steps: 131 | Train Loss: 0.3280267 Vali Loss: 0.6054561 Test Loss: 0.3791139
# EarlyStopping counter: 20 out of 100
# Updating learning rate to 9.847709021836118e-05
# >>>>>>>testing : test_ITSMixer_ETTm1_ftM_sl512_pl336_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 11185
# mse:0.3590959310531616, mae:0.38274115324020386, rse:0.5698880553245544