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
        self.drop = nn.Dropout(0.77)

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
        self.drop = nn.Dropout(0.34)

    def forward(self, x):
        res = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res

        x = self.drop(x)

        temp = x

        for i in range(3):
            self.layers[i] = x
            x = self.batchNorm2D(x)
            x = self.MLP_time1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time2(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = x + temp

        x = self.drop(x)

        for i in range(3):
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
# >>>>>>>start training : test_ITSMixer_ETTh1_ftM_sl512_pl96_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 8033
# val 2785
# test 2785
# Epoch: 1 cost time: 2.5945136547088623
# Epoch: 1, Steps: 31 | Train Loss: 0.7764309 Vali Loss: 1.1472021 Test Loss: 0.6578306
# Validation loss decreased (inf --> 1.147202).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 1.6164729595184326
# Epoch: 2, Steps: 31 | Train Loss: 0.5882139 Vali Loss: 0.6968220 Test Loss: 0.4541287
# Validation loss decreased (1.147202 --> 0.696822).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 1.6552786827087402
# Epoch: 3, Steps: 31 | Train Loss: 0.4666505 Vali Loss: 0.6431192 Test Loss: 0.4009285
# Validation loss decreased (0.696822 --> 0.643119).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 1.5871281623840332
# Epoch: 4, Steps: 31 | Train Loss: 0.4197807 Vali Loss: 0.6324152 Test Loss: 0.3834637
# Validation loss decreased (0.643119 --> 0.632415).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 2.1397011280059814
# Epoch: 5, Steps: 31 | Train Loss: 0.4012762 Vali Loss: 0.6245486 Test Loss: 0.3766473
# Validation loss decreased (0.632415 --> 0.624549).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 1.5999841690063477
# Epoch: 6, Steps: 31 | Train Loss: 0.3914798 Vali Loss: 0.6183082 Test Loss: 0.3729099
# Validation loss decreased (0.624549 --> 0.618308).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 1.6385796070098877
# Epoch: 7, Steps: 31 | Train Loss: 0.3839110 Vali Loss: 0.6263451 Test Loss: 0.3697414
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 1.592123031616211
# Epoch: 8, Steps: 31 | Train Loss: 0.3786435 Vali Loss: 0.6126870 Test Loss: 0.3688864
# Validation loss decreased (0.618308 --> 0.612687).  Saving model ...
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 1.7932238578796387
# Epoch: 9, Steps: 31 | Train Loss: 0.3754098 Vali Loss: 0.6189228 Test Loss: 0.3683404
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 1.6702885627746582
# Epoch: 10, Steps: 31 | Train Loss: 0.3705927 Vali Loss: 0.6130252 Test Loss: 0.3688931
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 1.5986425876617432
# Epoch: 11, Steps: 31 | Train Loss: 0.3694696 Vali Loss: 0.6084304 Test Loss: 0.3684455
# Validation loss decreased (0.612687 --> 0.608430).  Saving model ...
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 1.6188273429870605
# Epoch: 12, Steps: 31 | Train Loss: 0.3670731 Vali Loss: 0.6064336 Test Loss: 0.3663699
# Validation loss decreased (0.608430 --> 0.606434).  Saving model ...
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 1.6643083095550537
# Epoch: 13, Steps: 31 | Train Loss: 0.3652758 Vali Loss: 0.6110238 Test Loss: 0.3674722
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 2.2234678268432617
# Epoch: 14, Steps: 31 | Train Loss: 0.3636903 Vali Loss: 0.6118439 Test Loss: 0.3681721
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 1.5872085094451904
# Epoch: 15, Steps: 31 | Train Loss: 0.3610177 Vali Loss: 0.6187490 Test Loss: 0.3682984
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0002824295364810001
# Epoch: 16 cost time: 2.213355302810669
# Epoch: 16, Steps: 31 | Train Loss: 0.3596103 Vali Loss: 0.6148974 Test Loss: 0.3680917
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0002541865828329001
# Epoch: 17 cost time: 1.6286931037902832
# Epoch: 17, Steps: 31 | Train Loss: 0.3582225 Vali Loss: 0.6111173 Test Loss: 0.3680987
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002287679245496101
# Epoch: 18 cost time: 2.1076226234436035
# Epoch: 18, Steps: 31 | Train Loss: 0.3563435 Vali Loss: 0.6213648 Test Loss: 0.3678921
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0002058911320946491
# Epoch: 19 cost time: 1.5989429950714111
# Epoch: 19, Steps: 31 | Train Loss: 0.3556691 Vali Loss: 0.6132888 Test Loss: 0.3680089
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.00018530201888518417
# Epoch: 20 cost time: 1.6151213645935059
# Epoch: 20, Steps: 31 | Train Loss: 0.3552213 Vali Loss: 0.6203341 Test Loss: 0.3697873
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.00016677181699666576
# Epoch: 21 cost time: 1.615938425064087
# Epoch: 21, Steps: 31 | Train Loss: 0.3545827 Vali Loss: 0.6140511 Test Loss: 0.3681282
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.00015009463529699917
# Epoch: 22 cost time: 2.3306026458740234
# Epoch: 22, Steps: 31 | Train Loss: 0.3537425 Vali Loss: 0.6183690 Test Loss: 0.3682705
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.0001350851717672993
# Epoch: 23 cost time: 1.6257994174957275
# Epoch: 23, Steps: 31 | Train Loss: 0.3520354 Vali Loss: 0.6165427 Test Loss: 0.3686091
# EarlyStopping counter: 11 out of 100
# Updating learning rate to 0.00012157665459056935
# Epoch: 24 cost time: 1.6304473876953125
# Epoch: 24, Steps: 31 | Train Loss: 0.3520747 Vali Loss: 0.6199764 Test Loss: 0.3683206
# EarlyStopping counter: 12 out of 100
# Updating learning rate to 0.00010941898913151242
# Epoch: 25 cost time: 1.5977563858032227
# Epoch: 25, Steps: 31 | Train Loss: 0.3513654 Vali Loss: 0.6179289 Test Loss: 0.3695886
# EarlyStopping counter: 13 out of 100
# Updating learning rate to 9.847709021836118e-05
# Epoch: 26 cost time: 2.222698926925659
# Epoch: 26, Steps: 31 | Train Loss: 0.3505953 Vali Loss: 0.6220062 Test Loss: 0.3692701
# EarlyStopping counter: 14 out of 100
# Updating learning rate to 8.862938119652506e-05
# Epoch: 27 cost time: 1.6237571239471436
# Epoch: 27, Steps: 31 | Train Loss: 0.3504621 Vali Loss: 0.6221302 Test Loss: 0.3694820
# EarlyStopping counter: 15 out of 100
# Updating learning rate to 7.976644307687256e-05
# Epoch: 28 cost time: 1.6003518104553223
# Epoch: 28, Steps: 31 | Train Loss: 0.3500483 Vali Loss: 0.6231591 Test Loss: 0.3696202
# EarlyStopping counter: 16 out of 100
# Updating learning rate to 7.17897987691853e-05
# Epoch: 29 cost time: 1.6145784854888916
# Epoch: 29, Steps: 31 | Train Loss: 0.3500864 Vali Loss: 0.6183115 Test Loss: 0.3688299
# EarlyStopping counter: 17 out of 100
# Updating learning rate to 6.461081889226677e-05
# Epoch: 30 cost time: 1.8052217960357666
# Epoch: 30, Steps: 31 | Train Loss: 0.3492381 Vali Loss: 0.6155011 Test Loss: 0.3695544
# EarlyStopping counter: 18 out of 100
# Updating learning rate to 5.8149737003040094e-05
# Epoch: 31 cost time: 1.651291847229004
# Epoch: 31, Steps: 31 | Train Loss: 0.3494389 Vali Loss: 0.6207817 Test Loss: 0.3697764
# EarlyStopping counter: 19 out of 100
# Updating learning rate to 5.233476330273609e-05
# Epoch: 32 cost time: 1.620577096939087
# Epoch: 32, Steps: 31 | Train Loss: 0.3490304 Vali Loss: 0.6232908 Test Loss: 0.3699769
# EarlyStopping counter: 20 out of 100
# Updating learning rate to 4.7101286972462485e-05
# Epoch: 33 cost time: 1.6659116744995117
# Epoch: 33, Steps: 31 | Train Loss: 0.3488550 Vali Loss: 0.6253883 Test Loss: 0.3692690
# EarlyStopping counter: 21 out of 100
# Updating learning rate to 4.239115827521624e-05
# Epoch: 34 cost time: 1.716411828994751
# Epoch: 34, Steps: 31 | Train Loss: 0.3485864 Vali Loss: 0.6247553 Test Loss: 0.3701226
# EarlyStopping counter: 22 out of 100
# Updating learning rate to 3.8152042447694614e-05
# Epoch: 35 cost time: 1.771820306777954
# Epoch: 35, Steps: 31 | Train Loss: 0.3483714 Vali Loss: 0.6263014 Test Loss: 0.3701572
# EarlyStopping counter: 23 out of 100
# Updating learning rate to 3.433683820292515e-05
# >>>>>>>testing : test_ITSMixer_ETTh1_ftM_sl512_pl96_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2785
# mse:0.3482007682323456, mae:0.38453909754753113, rse:0.5619133710861206