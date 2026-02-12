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
        self.layers = [0] * 20
        self.drop = nn.Dropout(0.48)

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

        x = x + temp

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
# >>>>>>>start training : test_ITSMixer_ETTh2_ftM_sl512_pl192_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 7937
# val 2689
# test 2689
# Epoch: 1 cost time: 3.884483814239502
# Epoch: 1, Steps: 31 | Train Loss: 0.8923149 Vali Loss: 0.4739513 Test Loss: 0.4315530
# Validation loss decreased (inf --> 0.473951).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 4.0817766189575195
# Epoch: 2, Steps: 31 | Train Loss: 0.6596163 Vali Loss: 0.3565608 Test Loss: 0.3571689
# Validation loss decreased (0.473951 --> 0.356561).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 3.567507266998291
# Epoch: 3, Steps: 31 | Train Loss: 0.5361381 Vali Loss: 0.3415270 Test Loss: 0.3422902
# Validation loss decreased (0.356561 --> 0.341527).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 3.6867642402648926
# Epoch: 4, Steps: 31 | Train Loss: 0.4871915 Vali Loss: 0.3362364 Test Loss: 0.3327377
# Validation loss decreased (0.341527 --> 0.336236).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 3.6095244884490967
# Epoch: 5, Steps: 31 | Train Loss: 0.4649116 Vali Loss: 0.3429952 Test Loss: 0.3337672
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 3.6299235820770264
# Epoch: 6, Steps: 31 | Train Loss: 0.4528322 Vali Loss: 0.3451764 Test Loss: 0.3343368
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 3.8218014240264893
# Epoch: 7, Steps: 31 | Train Loss: 0.4420060 Vali Loss: 0.3583274 Test Loss: 0.3371904
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 3.6879353523254395
# Epoch: 8, Steps: 31 | Train Loss: 0.4325591 Vali Loss: 0.3480797 Test Loss: 0.3408493
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 4.255662441253662
# Epoch: 9, Steps: 31 | Train Loss: 0.4254658 Vali Loss: 0.3667848 Test Loss: 0.3421586
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 3.713287353515625
# Epoch: 10, Steps: 31 | Train Loss: 0.4191876 Vali Loss: 0.3730733 Test Loss: 0.3438102
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 3.762834072113037
# Epoch: 11, Steps: 31 | Train Loss: 0.4192527 Vali Loss: 0.3496871 Test Loss: 0.3453892
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 3.6290860176086426
# Epoch: 12, Steps: 31 | Train Loss: 0.4148378 Vali Loss: 0.3718947 Test Loss: 0.3580295
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 3.623162031173706
# Epoch: 13, Steps: 31 | Train Loss: 0.4094754 Vali Loss: 0.3609714 Test Loss: 0.3574473
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 3.762693405151367
# Epoch: 14, Steps: 31 | Train Loss: 0.4057042 Vali Loss: 0.3736131 Test Loss: 0.3573821
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 3.6022651195526123
# Epoch: 15, Steps: 31 | Train Loss: 0.4026829 Vali Loss: 0.3663007 Test Loss: 0.3550844
# EarlyStopping counter: 11 out of 100
# Updating learning rate to 0.0002824295364810001
# Epoch: 16 cost time: 4.125222682952881
# Epoch: 16, Steps: 31 | Train Loss: 0.3994338 Vali Loss: 0.3599719 Test Loss: 0.3573307
# EarlyStopping counter: 12 out of 100
# Updating learning rate to 0.0002541865828329001
# Epoch: 17 cost time: 3.6028976440429688
# Epoch: 17, Steps: 31 | Train Loss: 0.3961260 Vali Loss: 0.3720409 Test Loss: 0.3629251
# EarlyStopping counter: 13 out of 100
# Updating learning rate to 0.0002287679245496101
# Epoch: 18 cost time: 3.719414472579956
# Epoch: 18, Steps: 31 | Train Loss: 0.3943748 Vali Loss: 0.3777254 Test Loss: 0.3615710
# EarlyStopping counter: 14 out of 100
# Updating learning rate to 0.0002058911320946491
# Epoch: 19 cost time: 3.6299028396606445
# Epoch: 19, Steps: 31 | Train Loss: 0.3916055 Vali Loss: 0.3786071 Test Loss: 0.3651272
# EarlyStopping counter: 15 out of 100
# Updating learning rate to 0.00018530201888518417
# Epoch: 20 cost time: 3.6611359119415283
# Epoch: 20, Steps: 31 | Train Loss: 0.3907200 Vali Loss: 0.3629570 Test Loss: 0.3672547
# EarlyStopping counter: 16 out of 100
# Updating learning rate to 0.00016677181699666576
# Epoch: 21 cost time: 3.7478363513946533
# Epoch: 21, Steps: 31 | Train Loss: 0.3941473 Vali Loss: 0.3646491 Test Loss: 0.3648050
# EarlyStopping counter: 17 out of 100
# Updating learning rate to 0.00015009463529699917
# Epoch: 22 cost time: 3.7478418350219727
# Epoch: 22, Steps: 31 | Train Loss: 0.3917252 Vali Loss: 0.3624372 Test Loss: 0.3651175
# EarlyStopping counter: 18 out of 100
# Updating learning rate to 0.0001350851717672993
# Epoch: 23 cost time: 4.197859525680542
# Epoch: 23, Steps: 31 | Train Loss: 0.3884205 Vali Loss: 0.3657335 Test Loss: 0.3680086
# EarlyStopping counter: 19 out of 100
# Updating learning rate to 0.00012157665459056935
# Epoch: 24 cost time: 3.6830854415893555
# Epoch: 24, Steps: 31 | Train Loss: 0.3874459 Vali Loss: 0.3623653 Test Loss: 0.3726131
# EarlyStopping counter: 20 out of 100
# Updating learning rate to 0.00010941898913151242
# Epoch: 25 cost time: 3.765620231628418
# Epoch: 25, Steps: 31 | Train Loss: 0.3862807 Vali Loss: 0.3684886 Test Loss: 0.3690374
# EarlyStopping counter: 21 out of 100
# Updating learning rate to 9.847709021836118e-05
# Epoch: 26 cost time: 3.666395425796509
# Epoch: 26, Steps: 31 | Train Loss: 0.3841344 Vali Loss: 0.3676687 Test Loss: 0.3713589
# EarlyStopping counter: 22 out of 100
# Updating learning rate to 8.862938119652506e-05
# Epoch: 27 cost time: 3.700693130493164
# Epoch: 27, Steps: 31 | Train Loss: 0.3843492 Vali Loss: 0.3670970 Test Loss: 0.3729565
# EarlyStopping counter: 23 out of 100
# Updating learning rate to 7.976644307687256e-05
# Epoch: 28 cost time: 3.7755370140075684
# Epoch: 28, Steps: 31 | Train Loss: 0.3822875 Vali Loss: 0.3596701 Test Loss: 0.3696684
# EarlyStopping counter: 24 out of 100
# Updating learning rate to 7.17897987691853e-05
# Epoch: 29 cost time: 3.6321184635162354
# Epoch: 29, Steps: 31 | Train Loss: 0.3822092 Vali Loss: 0.3676074 Test Loss: 0.3689197
# EarlyStopping counter: 25 out of 100
# Updating learning rate to 6.461081889226677e-05
# Epoch: 30 cost time: 4.171072244644165
# Epoch: 30, Steps: 31 | Train Loss: 0.3817032 Vali Loss: 0.3570732 Test Loss: 0.3727847
# EarlyStopping counter: 26 out of 100
# Updating learning rate to 5.8149737003040094e-05
# Epoch: 31 cost time: 3.6525216102600098
# Epoch: 31, Steps: 31 | Train Loss: 0.3820481 Vali Loss: 0.3563284 Test Loss: 0.3751328
# EarlyStopping counter: 27 out of 100
# Updating learning rate to 5.233476330273609e-05
# Epoch: 32 cost time: 3.7425537109375
# Epoch: 32, Steps: 31 | Train Loss: 0.3818009 Vali Loss: 0.3617007 Test Loss: 0.3700374
# EarlyStopping counter: 28 out of 100
# Updating learning rate to 4.7101286972462485e-05
# Epoch: 33 cost time: 3.633107900619507
# Epoch: 33, Steps: 31 | Train Loss: 0.3816316 Vali Loss: 0.3631229 Test Loss: 0.3701329
# EarlyStopping counter: 29 out of 100
# Updating learning rate to 4.239115827521624e-05
# Epoch: 34 cost time: 3.7024126052856445
# Epoch: 34, Steps: 31 | Train Loss: 0.3807922 Vali Loss: 0.3577598 Test Loss: 0.3707584
# EarlyStopping counter: 30 out of 100
# Updating learning rate to 3.8152042447694614e-05
# Epoch: 35 cost time: 3.918109178543091
# Epoch: 35, Steps: 31 | Train Loss: 0.3802272 Vali Loss: 0.3562244 Test Loss: 0.3747426
# EarlyStopping counter: 31 out of 100
# Updating learning rate to 3.433683820292515e-05
# >>>>>>>testing : test_ITSMixer_ETTh2_ftM_sl512_pl192_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2689
# mse:0.3019455671310425, mae:0.36352989077568054, rse:0.438032329082489