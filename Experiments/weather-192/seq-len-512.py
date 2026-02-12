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
        self.drop = nn.Dropout(0.34)

    def forward(self, x):
        res = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res

        x = self.drop(x)
        temp = x

        for i in range(7):
            self.layers[i] = x
            x = self.batchNorm2D(x)
            x = self.MLP_time1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time2(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = temp + x

        x = self.drop(x)

        for i in range(7):
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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl192_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 36184
# val 5079
# test 10348
# 	iters: 100, epoch: 1 | loss: 0.6353039
# 	speed: 0.2091s/iter; left time: 569.0176s
# Epoch: 1 cost time: 27.973085641860962
# Epoch: 1, Steps: 141 | Train Loss: 0.7679614 Vali Loss: 0.5200604 Test Loss: 0.2937523
# Validation loss decreased (inf --> 0.520060).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.4762898
# 	speed: 0.3619s/iter; left time: 933.8224s
# Epoch: 2 cost time: 29.09566330909729
# Epoch: 2, Steps: 141 | Train Loss: 0.4894083 Vali Loss: 0.4132001 Test Loss: 0.2357204
# Validation loss decreased (0.520060 --> 0.413200).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.4470182
# 	speed: 0.3724s/iter; left time: 908.2195s
# Epoch: 3 cost time: 30.16904854774475
# Epoch: 3, Steps: 141 | Train Loss: 0.4310306 Vali Loss: 0.4012029 Test Loss: 0.2265184
# Validation loss decreased (0.413200 --> 0.401203).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.4035378
# 	speed: 0.3757s/iter; left time: 863.2460s
# Epoch: 4 cost time: 29.884673833847046
# Epoch: 4, Steps: 141 | Train Loss: 0.4197287 Vali Loss: 0.3992147 Test Loss: 0.2233630
# Validation loss decreased (0.401203 --> 0.399215).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.4263063
# 	speed: 0.3694s/iter; left time: 796.7538s
# Epoch: 5 cost time: 29.691259384155273
# Epoch: 5, Steps: 141 | Train Loss: 0.4141275 Vali Loss: 0.3908707 Test Loss: 0.2194200
# Validation loss decreased (0.399215 --> 0.390871).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.4350668
# 	speed: 0.3712s/iter; left time: 748.3100s
# Epoch: 6 cost time: 29.578386545181274
# Epoch: 6, Steps: 141 | Train Loss: 0.4094302 Vali Loss: 0.3892397 Test Loss: 0.2158986
# Validation loss decreased (0.390871 --> 0.389240).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.4197177
# 	speed: 0.3676s/iter; left time: 689.2022s
# Epoch: 7 cost time: 29.414331436157227
# Epoch: 7, Steps: 141 | Train Loss: 0.4051313 Vali Loss: 0.3822677 Test Loss: 0.2149391
# Validation loss decreased (0.389240 --> 0.382268).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.4862122
# 	speed: 0.3680s/iter; left time: 638.1542s
# Epoch: 8 cost time: 29.50326657295227
# Epoch: 8, Steps: 141 | Train Loss: 0.4024732 Vali Loss: 0.3830138 Test Loss: 0.2149591
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.4335395
# 	speed: 0.3686s/iter; left time: 587.2287s
# Epoch: 9 cost time: 29.401980876922607
# Epoch: 9, Steps: 141 | Train Loss: 0.3992702 Vali Loss: 0.3800177 Test Loss: 0.2140998
# Validation loss decreased (0.382268 --> 0.380018).  Saving model ...
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.3736129
# 	speed: 0.3728s/iter; left time: 541.3512s
# Epoch: 10 cost time: 29.492319583892822
# Epoch: 10, Steps: 141 | Train Loss: 0.3976227 Vali Loss: 0.3800453 Test Loss: 0.2126896
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.3616565
# 	speed: 0.3753s/iter; left time: 491.9994s
# Epoch: 11 cost time: 29.705881118774414
# Epoch: 11, Steps: 141 | Train Loss: 0.3952956 Vali Loss: 0.3759434 Test Loss: 0.2128482
# Validation loss decreased (0.380018 --> 0.375943).  Saving model ...
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.3937841
# 	speed: 0.3732s/iter; left time: 436.6294s
# Epoch: 12 cost time: 29.468177318572998
# Epoch: 12, Steps: 141 | Train Loss: 0.3938454 Vali Loss: 0.3783484 Test Loss: 0.2144624
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.4034715
# 	speed: 0.3727s/iter; left time: 383.4625s
# Epoch: 13 cost time: 29.41633701324463
# Epoch: 13, Steps: 141 | Train Loss: 0.3931536 Vali Loss: 0.3751700 Test Loss: 0.2126475
# Validation loss decreased (0.375943 --> 0.375170).  Saving model ...
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.3921162
# 	speed: 0.3751s/iter; left time: 333.0525s
# Epoch: 14 cost time: 29.523653030395508
# Epoch: 14, Steps: 141 | Train Loss: 0.3919719 Vali Loss: 0.3777053 Test Loss: 0.2132463
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.4139369
# 	speed: 0.3733s/iter; left time: 278.8404s
# Epoch: 15 cost time: 29.319823265075684
# Epoch: 15, Steps: 141 | Train Loss: 0.3905769 Vali Loss: 0.3761947 Test Loss: 0.2143263
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0002824295364810001
# 	iters: 100, epoch: 16 | loss: 0.4054370
# 	speed: 0.3721s/iter; left time: 225.4698s
# Epoch: 16 cost time: 29.331868648529053
# Epoch: 16, Steps: 141 | Train Loss: 0.3895824 Vali Loss: 0.3755136 Test Loss: 0.2116582
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0002541865828329001
# 	iters: 100, epoch: 17 | loss: 0.3263467
# 	speed: 0.3732s/iter; left time: 173.5494s
# Epoch: 17 cost time: 29.313920974731445
# Epoch: 17, Steps: 141 | Train Loss: 0.3885361 Vali Loss: 0.3774717 Test Loss: 0.2141600
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0002287679245496101
# 	iters: 100, epoch: 18 | loss: 0.3538249
# 	speed: 0.3724s/iter; left time: 120.6515s
# Epoch: 18 cost time: 29.405434131622314
# Epoch: 18, Steps: 141 | Train Loss: 0.3878333 Vali Loss: 0.3784129 Test Loss: 0.2148738
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002058911320946491
# 	iters: 100, epoch: 19 | loss: 0.4390879
# 	speed: 0.3731s/iter; left time: 68.2859s
# Epoch: 19 cost time: 29.299917936325073
# Epoch: 19, Steps: 141 | Train Loss: 0.3868200 Vali Loss: 0.3747098 Test Loss: 0.2121907
# Validation loss decreased (0.375170 --> 0.374710).  Saving model ...
# Updating learning rate to 0.00018530201888518417
# 	iters: 100, epoch: 20 | loss: 0.4243231
# 	speed: 0.3752s/iter; left time: 15.7580s
# Epoch: 20 cost time: 29.40628671646118
# Epoch: 20, Steps: 141 | Train Loss: 0.3858725 Vali Loss: 0.3777201 Test Loss: 0.2116944
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.00016677181699666576
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl192_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 10348
# mse:0.19054356217384338, mae:0.23383745551109314, rse:0.5742657780647278