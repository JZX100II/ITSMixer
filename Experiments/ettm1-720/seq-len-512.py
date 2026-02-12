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
        self.drop = nn.Dropout(0.33)

    def forward(self, x):
        res = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res

        x = self.drop(x)
        temp = x

        for i in range(6):
            self.layers[i] = x
            x = self.batchNorm2D(x)
            x = self.MLP_time1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time2(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = temp + x

        x = self.drop(x)

        for i in range(6):
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
# >>>>>>>start training : test_ITSMixer_ETTm1_ftM_sl512_pl720_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33329
# val 10801
# test 10801
# 	iters: 100, epoch: 1 | loss: 0.5850540
# 	speed: 0.0819s/iter; left time: 151.6195s
# Epoch: 1 cost time: 9.158066272735596
# Epoch: 1, Steps: 130 | Train Loss: 0.6504537 Vali Loss: 0.9163359 Test Loss: 0.5205517
# Validation loss decreased (inf --> 0.916336).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.4523627
# 	speed: 0.1439s/iter; left time: 247.5702s
# Epoch: 2 cost time: 8.984337568283081
# Epoch: 2, Steps: 130 | Train Loss: 0.4681401 Vali Loss: 0.8044732 Test Loss: 0.4197107
# Validation loss decreased (0.916336 --> 0.804473).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.4253356
# 	speed: 0.1480s/iter; left time: 235.4420s
# Epoch: 3 cost time: 9.15192198753357
# Epoch: 3, Steps: 130 | Train Loss: 0.4314813 Vali Loss: 0.8018964 Test Loss: 0.4159536
# Validation loss decreased (0.804473 --> 0.801896).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.4241552
# 	speed: 0.1476s/iter; left time: 215.6446s
# Epoch: 4 cost time: 8.975009441375732
# Epoch: 4, Steps: 130 | Train Loss: 0.4235924 Vali Loss: 0.8042803 Test Loss: 0.4150875
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.4140300
# 	speed: 0.1529s/iter; left time: 203.5687s
# Epoch: 5 cost time: 8.678438425064087
# Epoch: 5, Steps: 130 | Train Loss: 0.4191173 Vali Loss: 0.8001680 Test Loss: 0.4142715
# Validation loss decreased (0.801896 --> 0.800168).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.4221749
# 	speed: 0.1555s/iter; left time: 186.7843s
# Epoch: 6 cost time: 9.286231279373169
# Epoch: 6, Steps: 130 | Train Loss: 0.4156786 Vali Loss: 0.8072358 Test Loss: 0.4153991
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.4125234
# 	speed: 0.1526s/iter; left time: 163.4869s
# Epoch: 7 cost time: 9.337135314941406
# Epoch: 7, Steps: 130 | Train Loss: 0.4127947 Vali Loss: 0.8124751 Test Loss: 0.4160352
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.3978025
# 	speed: 0.1489s/iter; left time: 140.1557s
# Epoch: 8 cost time: 9.301537275314331
# Epoch: 8, Steps: 130 | Train Loss: 0.4096183 Vali Loss: 0.8274261 Test Loss: 0.4217232
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.4212094
# 	speed: 0.1493s/iter; left time: 121.0509s
# Epoch: 9 cost time: 9.030261516571045
# Epoch: 9, Steps: 130 | Train Loss: 0.4070912 Vali Loss: 0.8376392 Test Loss: 0.4233025
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3959168
# 	speed: 0.1473s/iter; left time: 100.3195s
# Epoch: 10 cost time: 8.528167247772217
# Epoch: 10, Steps: 130 | Train Loss: 0.4044318 Vali Loss: 0.8420821 Test Loss: 0.4293573
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.4048113
# 	speed: 0.1558s/iter; left time: 85.8647s
# Epoch: 11 cost time: 9.241130590438843
# Epoch: 11, Steps: 130 | Train Loss: 0.4021358 Vali Loss: 0.8519216 Test Loss: 0.4340010
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.3966098
# 	speed: 0.1511s/iter; left time: 63.6082s
# Epoch: 12 cost time: 9.311644554138184
# Epoch: 12, Steps: 130 | Train Loss: 0.3998727 Vali Loss: 0.8602247 Test Loss: 0.4367617
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.3998533
# 	speed: 0.1503s/iter; left time: 43.7381s
# Epoch: 13 cost time: 9.530810832977295
# Epoch: 13, Steps: 130 | Train Loss: 0.3981560 Vali Loss: 0.8683571 Test Loss: 0.4415616
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.3863537
# 	speed: 0.1490s/iter; left time: 23.9970s
# Epoch: 14 cost time: 8.925860166549683
# Epoch: 14, Steps: 130 | Train Loss: 0.3965786 Vali Loss: 0.8751453 Test Loss: 0.4435499
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.3784049
# 	speed: 0.1516s/iter; left time: 4.6991s
# Epoch: 15 cost time: 8.90341305732727
# Epoch: 15, Steps: 130 | Train Loss: 0.3947347 Vali Loss: 0.8832499 Test Loss: 0.4513125
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.0002824295364810001
# >>>>>>>testing : test_ITSMixer_ETTm1_ftM_sl512_pl720_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 10801
# mse:0.41552844643592834, mae:0.41301408410072327, rse:0.6131793260574341