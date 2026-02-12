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
        self.drop = nn.Dropout(0.29)

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
# >>>>>>>start training : test_ITSMixer_custom_ftM_sl512_pl720_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 35656
# val 4551
# test 9820
# 	iters: 100, epoch: 1 | loss: 0.7434795
# 	speed: 0.2241s/iter; left time: 476.1169s
# Epoch: 1 cost time: 29.557591915130615
# Epoch: 1, Steps: 139 | Train Loss: 0.8077102 Vali Loss: 0.6209579 Test Loss: 0.3707540
# Validation loss decreased (inf --> 0.620958).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.5507612
# 	speed: 0.4064s/iter; left time: 807.1068s
# Epoch: 2 cost time: 29.948283195495605
# Epoch: 2, Steps: 139 | Train Loss: 0.5700768 Vali Loss: 0.5425748 Test Loss: 0.3313434
# Validation loss decreased (0.620958 --> 0.542575).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.5202177
# 	speed: 0.4076s/iter; left time: 752.8774s
# Epoch: 3 cost time: 30.327281713485718
# Epoch: 3, Steps: 139 | Train Loss: 0.5218126 Vali Loss: 0.5384335 Test Loss: 0.3288001
# Validation loss decreased (0.542575 --> 0.538433).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.5335685
# 	speed: 0.4103s/iter; left time: 700.8150s
# Epoch: 4 cost time: 31.07120943069458
# Epoch: 4, Steps: 139 | Train Loss: 0.5150287 Vali Loss: 0.5340344 Test Loss: 0.3225923
# Validation loss decreased (0.538433 --> 0.534034).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.5651206
# 	speed: 0.4214s/iter; left time: 661.2269s
# Epoch: 5 cost time: 30.953881978988647
# Epoch: 5, Steps: 139 | Train Loss: 0.5107957 Vali Loss: 0.5350006 Test Loss: 0.3223777
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.5111448
# 	speed: 0.4164s/iter; left time: 595.4669s
# Epoch: 6 cost time: 30.980464220046997
# Epoch: 6, Steps: 139 | Train Loss: 0.5075104 Vali Loss: 0.5396975 Test Loss: 0.3225286
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.4780482
# 	speed: 0.4159s/iter; left time: 536.8998s
# Epoch: 7 cost time: 30.984357118606567
# Epoch: 7, Steps: 139 | Train Loss: 0.5046276 Vali Loss: 0.5364402 Test Loss: 0.3219285
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.5105945
# 	speed: 0.4197s/iter; left time: 483.5468s
# Epoch: 8 cost time: 31.257570505142212
# Epoch: 8, Steps: 139 | Train Loss: 0.5017351 Vali Loss: 0.5353928 Test Loss: 0.3214022
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.4910762
# 	speed: 0.4233s/iter; left time: 428.7567s
# Epoch: 9 cost time: 31.872499704360962
# Epoch: 9, Steps: 139 | Train Loss: 0.4988079 Vali Loss: 0.5291824 Test Loss: 0.3206191
# Validation loss decreased (0.534034 --> 0.529182).  Saving model ...
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.4757986
# 	speed: 0.4174s/iter; left time: 364.7939s
# Epoch: 10 cost time: 31.07961392402649
# Epoch: 10, Steps: 139 | Train Loss: 0.4967333 Vali Loss: 0.5306453 Test Loss: 0.3205922
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.4720054
# 	speed: 0.4141s/iter; left time: 304.3992s
# Epoch: 11 cost time: 31.130091190338135
# Epoch: 11, Steps: 139 | Train Loss: 0.4954013 Vali Loss: 0.5251903 Test Loss: 0.3207983
# Validation loss decreased (0.529182 --> 0.525190).  Saving model ...
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.5051494
# 	speed: 0.4176s/iter; left time: 248.8612s
# Epoch: 12 cost time: 31.640443563461304
# Epoch: 12, Steps: 139 | Train Loss: 0.4936716 Vali Loss: 0.5248377 Test Loss: 0.3189222
# Validation loss decreased (0.525190 --> 0.524838).  Saving model ...
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.4702772
# 	speed: 0.4226s/iter; left time: 193.1289s
# Epoch: 13 cost time: 31.166930198669434
# Epoch: 13, Steps: 139 | Train Loss: 0.4922906 Vali Loss: 0.5244792 Test Loss: 0.3228249
# Validation loss decreased (0.524838 --> 0.524479).  Saving model ...
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.4909602
# 	speed: 0.4184s/iter; left time: 133.0529s
# Epoch: 14 cost time: 31.24919080734253
# Epoch: 14, Steps: 139 | Train Loss: 0.4912039 Vali Loss: 0.5251675 Test Loss: 0.3201734
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.4637618
# 	speed: 0.4220s/iter; left time: 75.5466s
# Epoch: 15 cost time: 31.433948278427124
# Epoch: 15, Steps: 139 | Train Loss: 0.4901118 Vali Loss: 0.5260419 Test Loss: 0.3211990
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0002824295364810001
# 	iters: 100, epoch: 16 | loss: 0.5120518
# 	speed: 0.4176s/iter; left time: 16.7057s
# Epoch: 16 cost time: 31.140079259872437
# Epoch: 16, Steps: 139 | Train Loss: 0.4890807 Vali Loss: 0.5243573 Test Loss: 0.3212593
# Validation loss decreased (0.524479 --> 0.524357).  Saving model ...
# Updating learning rate to 0.0002541865828329001
# >>>>>>>testing : test_ITSMixer_custom_ftM_sl512_pl720_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 9820
# mse:0.31427133083343506, mae:0.3282468318939209, rse:0.7395540475845337