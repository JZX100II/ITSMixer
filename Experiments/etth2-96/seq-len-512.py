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
        self.drop = nn.Dropout(0.68)

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
# >>>>>>>start training : test_ITSMixer_ETTh2_ftM_sl512_pl96_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 8033
# val 2785
# test 2785
# Epoch: 1 cost time: 1.9556944370269775
# Epoch: 1, Steps: 31 | Train Loss: 0.9077306 Vali Loss: 0.4416078 Test Loss: 0.4126859
# Validation loss decreased (inf --> 0.441608).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 1.7025055885314941
# Epoch: 2, Steps: 31 | Train Loss: 0.6816528 Vali Loss: 0.3064906 Test Loss: 0.3204556
# Validation loss decreased (0.441608 --> 0.306491).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 1.6722352504730225
# Epoch: 3, Steps: 31 | Train Loss: 0.5029519 Vali Loss: 0.2791440 Test Loss: 0.3021714
# Validation loss decreased (0.306491 --> 0.279144).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 2.291572093963623
# Epoch: 4, Steps: 31 | Train Loss: 0.4351571 Vali Loss: 0.2710144 Test Loss: 0.2960082
# Validation loss decreased (0.279144 --> 0.271014).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 1.6583385467529297
# Epoch: 5, Steps: 31 | Train Loss: 0.4093483 Vali Loss: 0.2684497 Test Loss: 0.2942347
# Validation loss decreased (0.271014 --> 0.268450).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 1.6946818828582764
# Epoch: 6, Steps: 31 | Train Loss: 0.3997453 Vali Loss: 0.2736393 Test Loss: 0.2914803
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 1.6178584098815918
# Epoch: 7, Steps: 31 | Train Loss: 0.3920802 Vali Loss: 0.2723248 Test Loss: 0.2923643
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 2.2491800785064697
# Epoch: 8, Steps: 31 | Train Loss: 0.3860598 Vali Loss: 0.2729569 Test Loss: 0.2933168
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 1.6828718185424805
# Epoch: 9, Steps: 31 | Train Loss: 0.3785750 Vali Loss: 0.2744023 Test Loss: 0.2925862
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 1.654644250869751
# Epoch: 10, Steps: 31 | Train Loss: 0.3754848 Vali Loss: 0.2723225 Test Loss: 0.2918251
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 1.6572110652923584
# Epoch: 11, Steps: 31 | Train Loss: 0.3731196 Vali Loss: 0.2701586 Test Loss: 0.2938509
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 2.196949005126953
# Epoch: 12, Steps: 31 | Train Loss: 0.3694148 Vali Loss: 0.2684360 Test Loss: 0.2931607
# Validation loss decreased (0.268450 --> 0.268436).  Saving model ...
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 1.725024938583374
# Epoch: 13, Steps: 31 | Train Loss: 0.3685532 Vali Loss: 0.2729464 Test Loss: 0.2922267
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 1.674914836883545
# Epoch: 14, Steps: 31 | Train Loss: 0.3664705 Vali Loss: 0.2760276 Test Loss: 0.2917156
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 1.8201806545257568
# Epoch: 15, Steps: 31 | Train Loss: 0.3652134 Vali Loss: 0.2767715 Test Loss: 0.2915227
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0002824295364810001
# Epoch: 16 cost time: 2.3945419788360596
# Epoch: 16, Steps: 31 | Train Loss: 0.3625501 Vali Loss: 0.2799644 Test Loss: 0.2919799
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0002541865828329001
# Epoch: 17 cost time: 1.689770221710205
# Epoch: 17, Steps: 31 | Train Loss: 0.3596673 Vali Loss: 0.2787567 Test Loss: 0.2927444
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002287679245496101
# Epoch: 18 cost time: 1.6578264236450195
# Epoch: 18, Steps: 31 | Train Loss: 0.3573448 Vali Loss: 0.2751766 Test Loss: 0.2946756
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0002058911320946491
# Epoch: 19 cost time: 1.6597867012023926
# Epoch: 19, Steps: 31 | Train Loss: 0.3576557 Vali Loss: 0.2760977 Test Loss: 0.2939190
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.00018530201888518417
# Epoch: 20 cost time: 2.331397294998169
# Epoch: 20, Steps: 31 | Train Loss: 0.3559962 Vali Loss: 0.2744072 Test Loss: 0.2928139
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.00016677181699666576
# Epoch: 21 cost time: 1.6723511219024658
# Epoch: 21, Steps: 31 | Train Loss: 0.3561635 Vali Loss: 0.2731654 Test Loss: 0.2929413
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.00015009463529699917
# Epoch: 22 cost time: 1.7271208763122559
# Epoch: 22, Steps: 31 | Train Loss: 0.3548122 Vali Loss: 0.2773120 Test Loss: 0.2948597
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.0001350851717672993
# Epoch: 23 cost time: 1.6745765209197998
# Epoch: 23, Steps: 31 | Train Loss: 0.3533187 Vali Loss: 0.2787910 Test Loss: 0.2928682
# EarlyStopping counter: 11 out of 100
# Updating learning rate to 0.00012157665459056935
# Epoch: 24 cost time: 2.410728931427002
# Epoch: 24, Steps: 31 | Train Loss: 0.3547607 Vali Loss: 0.2727850 Test Loss: 0.2949234
# EarlyStopping counter: 12 out of 100
# Updating learning rate to 0.00010941898913151242
# Epoch: 25 cost time: 1.6816325187683105
# Epoch: 25, Steps: 31 | Train Loss: 0.3533888 Vali Loss: 0.2762788 Test Loss: 0.2937294
# EarlyStopping counter: 13 out of 100
# Updating learning rate to 9.847709021836118e-05
# >>>>>>>testing : test_ITSMixer_ETTh2_ftM_sl512_pl96_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2785
# mse:0.25948721170425415, mae:0.3268342614173889, rse:0.40675342082977295