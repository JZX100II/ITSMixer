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
        self.layers = [0] * 50
        self.drop = nn.Dropout(0.67)

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
# >>>>>>>start training : test_ITSMixer_ETTm2_ftM_sl512_pl336_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33713
# val 11185
# test 11185
# 	iters: 100, epoch: 1 | loss: 0.5736455
# 	speed: 0.0985s/iter; left time: 119.3437s
# Epoch: 1 cost time: 11.657837867736816
# Epoch: 1, Steps: 131 | Train Loss: 0.6479831 Vali Loss: 0.2971596 Test Loss: 0.3553171
# Validation loss decreased (inf --> 0.297160).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.3772925
# 	speed: 0.1687s/iter; left time: 182.1911s
# Epoch: 2 cost time: 11.883501529693604
# Epoch: 2, Steps: 131 | Train Loss: 0.4268969 Vali Loss: 0.2495998 Test Loss: 0.3016728
# Validation loss decreased (0.297160 --> 0.249600).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.3617936
# 	speed: 0.1763s/iter; left time: 167.3437s
# Epoch: 3 cost time: 12.029318571090698
# Epoch: 3, Steps: 131 | Train Loss: 0.3765947 Vali Loss: 0.2520081 Test Loss: 0.3117068
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.3428319
# 	speed: 0.1732s/iter; left time: 141.7132s
# Epoch: 4 cost time: 11.998748540878296
# Epoch: 4, Steps: 131 | Train Loss: 0.3663712 Vali Loss: 0.2729451 Test Loss: 0.3338020
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.3965091
# 	speed: 0.1736s/iter; left time: 119.2883s
# Epoch: 5 cost time: 12.175905227661133
# Epoch: 5, Steps: 131 | Train Loss: 0.3582796 Vali Loss: 0.2769724 Test Loss: 0.3361879
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.3477274
# 	speed: 0.1757s/iter; left time: 97.6744s
# Epoch: 6 cost time: 11.701963663101196
# Epoch: 6, Steps: 131 | Train Loss: 0.3523739 Vali Loss: 0.2692866 Test Loss: 0.3246724
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.3461397
# 	speed: 0.1687s/iter; left time: 71.7168s
# Epoch: 7 cost time: 11.785528182983398
# Epoch: 7, Steps: 131 | Train Loss: 0.3456839 Vali Loss: 0.2673630 Test Loss: 0.3287912
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.3202723
# 	speed: 0.1756s/iter; left time: 51.6149s
# Epoch: 8 cost time: 11.866875171661377
# Epoch: 8, Steps: 131 | Train Loss: 0.3449370 Vali Loss: 0.2740719 Test Loss: 0.3369194
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.3515891
# 	speed: 0.1713s/iter; left time: 27.9206s
# Epoch: 9 cost time: 11.926152229309082
# Epoch: 9, Steps: 131 | Train Loss: 0.3391078 Vali Loss: 0.2692172 Test Loss: 0.3332765
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3169818
# 	speed: 0.1748s/iter; left time: 5.5937s
# Epoch: 10 cost time: 12.1613028049469
# Epoch: 10, Steps: 131 | Train Loss: 0.3372157 Vali Loss: 0.2779696 Test Loss: 0.3444030
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0004782969000000001
# >>>>>>>testing : test_ITSMixer_ETTm2_ftM_sl512_pl336_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 11185
# mse:0.2752225399017334, mae:0.32812321186065674, rse:0.42204487323760986