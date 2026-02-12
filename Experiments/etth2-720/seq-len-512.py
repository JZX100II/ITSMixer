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
        self.MLP_time4 = Mlp_time(time_dim, time_dim)
        self.MLP_time5 = Mlp_time(time_dim, time_dim)
        self.layers = [0] * 50
        self.drop = nn.Dropout(0.64)

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

        x = x + temp

        x = self.drop(x)

        temp = x

        for i in range(8):
            x = self.batchNorm2D(x)
            x = self.MLP_time3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time4(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = x + temp

        x = self.drop(x)
        
        for i in range(8):
            x = self.batchNorm2D(x)
            x = self.MLP_time5(x.permute(0, 2, 1)).permute(0, 2, 1)
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
# >>>>>>>start training : test_ITSMixer_ETTh2_ftM_sl512_pl720_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 7409
# val 2161
# test 2161
# Epoch: 1 cost time: 4.765098333358765
# Epoch: 1, Steps: 28 | Train Loss: 1.1159460 Vali Loss: 0.7356467 Test Loss: 0.4705075
# Validation loss decreased (inf --> 0.735647).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 4.499833345413208
# Epoch: 2, Steps: 28 | Train Loss: 0.8992336 Vali Loss: 0.6389090 Test Loss: 0.4130302
# Validation loss decreased (0.735647 --> 0.638909).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 4.394993543624878
# Epoch: 3, Steps: 28 | Train Loss: 0.7564689 Vali Loss: 0.6219800 Test Loss: 0.4037951
# Validation loss decreased (0.638909 --> 0.621980).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 4.539931297302246
# Epoch: 4, Steps: 28 | Train Loss: 0.6927008 Vali Loss: 0.6228676 Test Loss: 0.4060428
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 4.421934127807617
# Epoch: 5, Steps: 28 | Train Loss: 0.6650401 Vali Loss: 0.6264998 Test Loss: 0.4121976
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 4.618331432342529
# Epoch: 6, Steps: 28 | Train Loss: 0.6535202 Vali Loss: 0.6255220 Test Loss: 0.4215289
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 4.650385856628418
# Epoch: 7, Steps: 28 | Train Loss: 0.6433494 Vali Loss: 0.6462964 Test Loss: 0.4145924
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 5.0457727909088135
# Epoch: 8, Steps: 28 | Train Loss: 0.6346691 Vali Loss: 0.6455842 Test Loss: 0.4244082
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 4.638273239135742
# Epoch: 9, Steps: 28 | Train Loss: 0.6250212 Vali Loss: 0.6639743 Test Loss: 0.4433715
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 4.7620530128479
# Epoch: 10, Steps: 28 | Train Loss: 0.6207480 Vali Loss: 0.6734496 Test Loss: 0.4434649
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 4.601541519165039
# Epoch: 11, Steps: 28 | Train Loss: 0.6156969 Vali Loss: 0.6738549 Test Loss: 0.4604664
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 4.720893383026123
# Epoch: 12, Steps: 28 | Train Loss: 0.6254548 Vali Loss: 0.6239414 Test Loss: 0.4476595
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 4.558174133300781
# Epoch: 13, Steps: 28 | Train Loss: 0.6316689 Vali Loss: 0.6291615 Test Loss: 0.4423394
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 4.770901203155518
# Epoch: 14, Steps: 28 | Train Loss: 0.6263780 Vali Loss: 0.6356673 Test Loss: 0.4454718
# EarlyStopping counter: 11 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 4.542198419570923
# Epoch: 15, Steps: 28 | Train Loss: 0.6177140 Vali Loss: 0.6383249 Test Loss: 0.4426521
# EarlyStopping counter: 12 out of 100
# Updating learning rate to 0.0002824295364810001
# Epoch: 16 cost time: 4.6821208000183105
# Epoch: 16, Steps: 28 | Train Loss: 0.6161710 Vali Loss: 0.6495949 Test Loss: 0.4526616
# EarlyStopping counter: 13 out of 100
# Updating learning rate to 0.0002541865828329001
# Epoch: 17 cost time: 4.461136102676392
# Epoch: 17, Steps: 28 | Train Loss: 0.6099020 Vali Loss: 0.6543531 Test Loss: 0.4546539
# EarlyStopping counter: 14 out of 100
# Updating learning rate to 0.0002287679245496101
# Epoch: 18 cost time: 4.623773097991943
# Epoch: 18, Steps: 28 | Train Loss: 0.6066974 Vali Loss: 0.6568601 Test Loss: 0.4681249
# EarlyStopping counter: 15 out of 100
# Updating learning rate to 0.0002058911320946491
# Epoch: 19 cost time: 4.458263158798218
# Epoch: 19, Steps: 28 | Train Loss: 0.6038301 Vali Loss: 0.6615001 Test Loss: 0.4635154
# EarlyStopping counter: 16 out of 100
# Updating learning rate to 0.00018530201888518417
# Epoch: 20 cost time: 4.610806703567505
# Epoch: 20, Steps: 28 | Train Loss: 0.6035434 Vali Loss: 0.6555862 Test Loss: 0.4637268
# EarlyStopping counter: 17 out of 100
# Updating learning rate to 0.00016677181699666576
# Epoch: 21 cost time: 5.630251169204712
# Epoch: 21, Steps: 28 | Train Loss: 0.6029060 Vali Loss: 0.6673535 Test Loss: 0.4641670
# EarlyStopping counter: 18 out of 100
# Updating learning rate to 0.00015009463529699917
# Epoch: 22 cost time: 5.303614377975464
# Epoch: 22, Steps: 28 | Train Loss: 0.6031042 Vali Loss: 0.6660162 Test Loss: 0.4597843
# EarlyStopping counter: 19 out of 100
# Updating learning rate to 0.0001350851717672993
# Epoch: 23 cost time: 4.569138765335083
# Epoch: 23, Steps: 28 | Train Loss: 0.6001137 Vali Loss: 0.6510390 Test Loss: 0.4503967
# EarlyStopping counter: 20 out of 100
# Updating learning rate to 0.00012157665459056935
# Epoch: 24 cost time: 4.799948692321777
# Epoch: 24, Steps: 28 | Train Loss: 0.6255082 Vali Loss: 0.6349249 Test Loss: 0.4555244
# EarlyStopping counter: 21 out of 100
# Updating learning rate to 0.00010941898913151242
# Epoch: 25 cost time: 4.537595272064209
# Epoch: 25, Steps: 28 | Train Loss: 0.6196734 Vali Loss: 0.6314052 Test Loss: 0.4549825
# EarlyStopping counter: 22 out of 100
# Updating learning rate to 9.847709021836118e-05
# >>>>>>>testing : test_ITSMixer_ETTh2_ftM_sl512_pl720_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2161
# mse:0.3814588189125061, mae:0.42613133788108826, rse:0.49516627192497253