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
        self.drop = nn.Dropout(0.81)

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
# >>>>>>>start training : test_ITSMixer_ETTh2_ftM_sl512_pl336_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 7793
# val 2545
# test 2545
# Epoch: 1 cost time: 4.7147908210754395
# Epoch: 1, Steps: 30 | Train Loss: 1.2577784 Vali Loss: 0.5551209 Test Loss: 0.4246564
# Validation loss decreased (inf --> 0.555121).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 4.4056010246276855
# Epoch: 2, Steps: 30 | Train Loss: 0.8969533 Vali Loss: 0.4367776 Test Loss: 0.3540043
# Validation loss decreased (0.555121 --> 0.436778).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 4.530538082122803
# Epoch: 3, Steps: 30 | Train Loss: 0.6386102 Vali Loss: 0.4257695 Test Loss: 0.3459972
# Validation loss decreased (0.436778 --> 0.425770).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 4.453812599182129
# Epoch: 4, Steps: 30 | Train Loss: 0.5622663 Vali Loss: 0.4212886 Test Loss: 0.3441490
# Validation loss decreased (0.425770 --> 0.421289).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 4.6111767292022705
# Epoch: 5, Steps: 30 | Train Loss: 0.5455124 Vali Loss: 0.4170079 Test Loss: 0.3455710
# Validation loss decreased (0.421289 --> 0.417008).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 4.520635366439819
# Epoch: 6, Steps: 30 | Train Loss: 0.5393270 Vali Loss: 0.4210473 Test Loss: 0.3465321
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 4.573241233825684
# Epoch: 7, Steps: 30 | Train Loss: 0.5363678 Vali Loss: 0.4252943 Test Loss: 0.3475827
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 4.535363435745239
# Epoch: 8, Steps: 30 | Train Loss: 0.5349828 Vali Loss: 0.4248590 Test Loss: 0.3467768
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 4.621194362640381
# Epoch: 9, Steps: 30 | Train Loss: 0.5333325 Vali Loss: 0.4257838 Test Loss: 0.3457594
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 4.515772581100464
# Epoch: 10, Steps: 30 | Train Loss: 0.5312688 Vali Loss: 0.4241799 Test Loss: 0.3469929
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0004782969000000001
# >>>>>>>testing : test_ITSMixer_ETTh2_ftM_sl512_pl336_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2545
# mse:0.3135843575000763, mae:0.3775578737258911, rse:0.44893133640289307