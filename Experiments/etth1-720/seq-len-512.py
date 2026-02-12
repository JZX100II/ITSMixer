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
        self.drop = nn.Dropout(0.81)

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
        self.layers = [0] * 10
        self.drop = nn.Dropout(0.66)

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
# >>>>>>>start training : test_ITSMixer_ETTh1_ftM_sl512_pl720_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 7409
# val 2161
# test 2161
# Epoch: 1 cost time: 4.835546970367432
# Epoch: 1, Steps: 28 | Train Loss: 1.2060949 Vali Loss: 1.5466516 Test Loss: 0.7234254
# Validation loss decreased (inf --> 1.546652).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 4.645474910736084
# Epoch: 2, Steps: 28 | Train Loss: 0.8701743 Vali Loss: 1.2087896 Test Loss: 0.4783025
# Validation loss decreased (1.546652 --> 1.208790).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 4.484739542007446
# Epoch: 3, Steps: 28 | Train Loss: 0.6621476 Vali Loss: 1.1433755 Test Loss: 0.4423940
# Validation loss decreased (1.208790 --> 1.143376).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 4.542235851287842
# Epoch: 4, Steps: 28 | Train Loss: 0.5920360 Vali Loss: 1.1234224 Test Loss: 0.4330938
# Validation loss decreased (1.143376 --> 1.123422).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 4.624326467514038
# Epoch: 5, Steps: 28 | Train Loss: 0.5697775 Vali Loss: 1.1144547 Test Loss: 0.4272204
# Validation loss decreased (1.123422 --> 1.114455).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 4.835586309432983
# Epoch: 6, Steps: 28 | Train Loss: 0.5607327 Vali Loss: 1.1088986 Test Loss: 0.4255332
# Validation loss decreased (1.114455 --> 1.108899).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 4.640557289123535
# Epoch: 7, Steps: 28 | Train Loss: 0.5553815 Vali Loss: 1.1069889 Test Loss: 0.4248474
# Validation loss decreased (1.108899 --> 1.106989).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 4.650599241256714
# Epoch: 8, Steps: 28 | Train Loss: 0.5524027 Vali Loss: 1.1062664 Test Loss: 0.4227437
# Validation loss decreased (1.106989 --> 1.106266).  Saving model ...
# Updating learning rate to 0.00059049
# >>>>>>>testing : test_ITSMixer_ETTh1_ftM_sl512_pl720_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2161
# mse:0.40666329860687256, mae:0.4388241767883301, rse:0.6124950051307678