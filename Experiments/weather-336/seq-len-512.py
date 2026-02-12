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
        self.drop = nn.Dropout(0.39)

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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl336_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 36040
# val 4935
# test 10204
# 	iters: 100, epoch: 1 | loss: 0.7628992
# 	speed: 0.2707s/iter; left time: 731.2671s
# Epoch: 1 cost time: 31.36426568031311
# Epoch: 1, Steps: 140 | Train Loss: 0.8219029 Vali Loss: 0.5683115 Test Loss: 0.3273267
# Validation loss decreased (inf --> 0.568312).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.4935917
# 	speed: 0.3988s/iter; left time: 1021.4127s
# Epoch: 2 cost time: 31.49494433403015
# Epoch: 2, Steps: 140 | Train Loss: 0.5315175 Vali Loss: 0.4713179 Test Loss: 0.2782676
# Validation loss decreased (0.568312 --> 0.471318).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.5044327
# 	speed: 0.4043s/iter; left time: 978.8514s
# Epoch: 3 cost time: 31.962181091308594
# Epoch: 3, Steps: 140 | Train Loss: 0.4737695 Vali Loss: 0.4649920 Test Loss: 0.2729940
# Validation loss decreased (0.471318 --> 0.464992).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.4738672
# 	speed: 0.4111s/iter; left time: 937.7743s
# Epoch: 4 cost time: 32.77445387840271
# Epoch: 4, Steps: 140 | Train Loss: 0.4647441 Vali Loss: 0.4564659 Test Loss: 0.2654969
# Validation loss decreased (0.464992 --> 0.456466).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.4905804
# 	speed: 0.4173s/iter; left time: 893.3929s
# Epoch: 5 cost time: 32.98531770706177
# Epoch: 5, Steps: 140 | Train Loss: 0.4601442 Vali Loss: 0.4568996 Test Loss: 0.2650431
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.4231526
# 	speed: 0.4212s/iter; left time: 842.9136s
# Epoch: 6 cost time: 33.71790075302124
# Epoch: 6, Steps: 140 | Train Loss: 0.4565830 Vali Loss: 0.4538400 Test Loss: 0.2634534
# Validation loss decreased (0.456466 --> 0.453840).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.4733275
# 	speed: 0.4223s/iter; left time: 785.8537s
# Epoch: 7 cost time: 33.5542049407959
# Epoch: 7, Steps: 140 | Train Loss: 0.4533976 Vali Loss: 0.4504976 Test Loss: 0.2608074
# Validation loss decreased (0.453840 --> 0.450498).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.4665259
# 	speed: 0.4225s/iter; left time: 727.1009s
# Epoch: 8 cost time: 33.717729568481445
# Epoch: 8, Steps: 140 | Train Loss: 0.4497641 Vali Loss: 0.4438399 Test Loss: 0.2575657
# Validation loss decreased (0.450498 --> 0.443840).  Saving model ...
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.4521653
# 	speed: 0.4266s/iter; left time: 674.4650s
# Epoch: 9 cost time: 33.84911012649536
# Epoch: 9, Steps: 140 | Train Loss: 0.4468217 Vali Loss: 0.4423300 Test Loss: 0.2574425
# Validation loss decreased (0.443840 --> 0.442330).  Saving model ...
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.4292006
# 	speed: 0.4230s/iter; left time: 609.5277s
# Epoch: 10 cost time: 33.79361319541931
# Epoch: 10, Steps: 140 | Train Loss: 0.4450774 Vali Loss: 0.4399872 Test Loss: 0.2573171
# Validation loss decreased (0.442330 --> 0.439987).  Saving model ...
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.4200434
# 	speed: 0.4273s/iter; left time: 555.9008s
# Epoch: 11 cost time: 33.93755865097046
# Epoch: 11, Steps: 140 | Train Loss: 0.4435528 Vali Loss: 0.4403777 Test Loss: 0.2565927
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.4007276
# 	speed: 0.4247s/iter; left time: 493.1266s
# Epoch: 12 cost time: 33.793805837631226
# Epoch: 12, Steps: 140 | Train Loss: 0.4416673 Vali Loss: 0.4388699 Test Loss: 0.2562647
# Validation loss decreased (0.439987 --> 0.438870).  Saving model ...
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.4203593
# 	speed: 0.4279s/iter; left time: 436.8981s
# Epoch: 13 cost time: 34.10020637512207
# Epoch: 13, Steps: 140 | Train Loss: 0.4405958 Vali Loss: 0.4355776 Test Loss: 0.2557225
# Validation loss decreased (0.438870 --> 0.435578).  Saving model ...
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.4143744
# 	speed: 0.4267s/iter; left time: 375.8964s
# Epoch: 14 cost time: 33.91866326332092
# Epoch: 14, Steps: 140 | Train Loss: 0.4391657 Vali Loss: 0.4375829 Test Loss: 0.2561555
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.4593228
# 	speed: 0.4248s/iter; left time: 314.8057s
# Epoch: 15 cost time: 34.01428008079529
# Epoch: 15, Steps: 140 | Train Loss: 0.4382385 Vali Loss: 0.4343841 Test Loss: 0.2556736
# Validation loss decreased (0.435578 --> 0.434384).  Saving model ...
# Updating learning rate to 0.0002824295364810001
# 	iters: 100, epoch: 16 | loss: 0.4833659
# 	speed: 0.4288s/iter; left time: 257.6808s
# Epoch: 16 cost time: 33.87347984313965
# Epoch: 16, Steps: 140 | Train Loss: 0.4374263 Vali Loss: 0.4357025 Test Loss: 0.2569293
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0002541865828329001
# 	iters: 100, epoch: 17 | loss: 0.4373751
# 	speed: 0.4234s/iter; left time: 195.2015s
# Epoch: 17 cost time: 33.90216588973999
# Epoch: 17, Steps: 140 | Train Loss: 0.4363495 Vali Loss: 0.4347692 Test Loss: 0.2566940
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0002287679245496101
# 	iters: 100, epoch: 18 | loss: 0.4172665
# 	speed: 0.4281s/iter; left time: 137.4088s
# Epoch: 18 cost time: 33.89305925369263
# Epoch: 18, Steps: 140 | Train Loss: 0.4354644 Vali Loss: 0.4349723 Test Loss: 0.2558392
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0002058911320946491
# 	iters: 100, epoch: 19 | loss: 0.4432608
# 	speed: 0.4249s/iter; left time: 76.9121s
# Epoch: 19 cost time: 34.05927109718323
# Epoch: 19, Steps: 140 | Train Loss: 0.4347417 Vali Loss: 0.4367203 Test Loss: 0.2570485
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.00018530201888518417
# 	iters: 100, epoch: 20 | loss: 0.4178148
# 	speed: 0.4294s/iter; left time: 17.6034s
# Epoch: 20 cost time: 33.990336894989014
# Epoch: 20, Steps: 140 | Train Loss: 0.4339515 Vali Loss: 0.4369051 Test Loss: 0.2573166
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.00016677181699666576
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl336_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 10204
# mse:0.2385307401418686, mae:0.27281656861305237, rse:0.6436973810195923