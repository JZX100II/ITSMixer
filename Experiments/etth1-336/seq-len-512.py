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
        self.drop = nn.Dropout(0.63)

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

        x = x + temp

        x = self.drop(x)

        temp = x

        for i in range(6):
            x = self.batchNorm2D(x)
            x = self.MLP_time3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.layers[i]

            temp = self.batchNorm2D(temp)
            temp = self.MLP_time4(temp.permute(0, 2, 1)).permute(0, 2, 1)
            temp = temp + self.layers[i]

        x = x + temp

        x = self.drop(x)
        
        for i in range(6):
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
# >>>>>>>start training : test_ITSMixer_ETTh1_ftM_sl512_pl336_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 7793
# val 2545
# test 2545
# Epoch: 1 cost time: 3.889307737350464
# Epoch: 1, Steps: 30 | Train Loss: 1.0357690 Vali Loss: 1.3541842 Test Loss: 0.6837415
# Validation loss decreased (inf --> 1.354184).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 3.6463053226470947
# Epoch: 2, Steps: 30 | Train Loss: 0.7587027 Vali Loss: 1.0449884 Test Loss: 0.4633519
# Validation loss decreased (1.354184 --> 1.044988).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 3.6080267429351807
# Epoch: 3, Steps: 30 | Train Loss: 0.5711038 Vali Loss: 0.9850276 Test Loss: 0.4146083
# Validation loss decreased (1.044988 --> 0.985028).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 3.7196767330169678
# Epoch: 4, Steps: 30 | Train Loss: 0.5040136 Vali Loss: 0.9572691 Test Loss: 0.4003237
# Validation loss decreased (0.985028 --> 0.957269).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 3.650843620300293
# Epoch: 5, Steps: 30 | Train Loss: 0.4818466 Vali Loss: 0.9669613 Test Loss: 0.3933844
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 3.7094602584838867
# Epoch: 6, Steps: 30 | Train Loss: 0.4731062 Vali Loss: 0.9743684 Test Loss: 0.3934047
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 3.813089609146118
# Epoch: 7, Steps: 30 | Train Loss: 0.4691196 Vali Loss: 0.9780114 Test Loss: 0.3952719
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 3.7227931022644043
# Epoch: 8, Steps: 30 | Train Loss: 0.4665539 Vali Loss: 0.9752970 Test Loss: 0.3920035
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 4.481284141540527
# Epoch: 9, Steps: 30 | Train Loss: 0.4646602 Vali Loss: 0.9802552 Test Loss: 0.3920789
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 3.6782195568084717
# Epoch: 10, Steps: 30 | Train Loss: 0.4625474 Vali Loss: 0.9756246 Test Loss: 0.3924064
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 3.883140802383423
# Epoch: 11, Steps: 30 | Train Loss: 0.4616411 Vali Loss: 0.9679294 Test Loss: 0.3922252
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 3.762805223464966
# Epoch: 12, Steps: 30 | Train Loss: 0.4609992 Vali Loss: 0.9605139 Test Loss: 0.3918371
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 3.844818115234375
# Epoch: 13, Steps: 30 | Train Loss: 0.4590669 Vali Loss: 0.9572401 Test Loss: 0.3942550
# Validation loss decreased (0.957269 --> 0.957240).  Saving model ...
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 3.8504607677459717
# Epoch: 14, Steps: 30 | Train Loss: 0.4567533 Vali Loss: 0.9587197 Test Loss: 0.3968784
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 3.7876999378204346
# Epoch: 15, Steps: 30 | Train Loss: 0.4552725 Vali Loss: 0.9605643 Test Loss: 0.3927087
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0002824295364810001
# Epoch: 16 cost time: 4.4436867237091064
# Epoch: 16, Steps: 30 | Train Loss: 0.4536990 Vali Loss: 0.9612995 Test Loss: 0.3948286
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0002541865828329001
# Epoch: 17 cost time: 3.80535888671875
# Epoch: 17, Steps: 30 | Train Loss: 0.4528911 Vali Loss: 0.9672695 Test Loss: 0.3920351
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0002287679245496101
# Epoch: 18 cost time: 3.8256194591522217
# Epoch: 18, Steps: 30 | Train Loss: 0.4509229 Vali Loss: 0.9645707 Test Loss: 0.3968441
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002058911320946491
# Epoch: 19 cost time: 3.725343942642212
# Epoch: 19, Steps: 30 | Train Loss: 0.4510411 Vali Loss: 0.9661341 Test Loss: 0.3986778
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.00018530201888518417
# Epoch: 20 cost time: 3.7938778400421143
# Epoch: 20, Steps: 30 | Train Loss: 0.4495795 Vali Loss: 0.9646901 Test Loss: 0.3966714
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.00016677181699666576
# Epoch: 21 cost time: 3.7936339378356934
# Epoch: 21, Steps: 30 | Train Loss: 0.4483427 Vali Loss: 0.9638811 Test Loss: 0.4039497
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.00015009463529699917
# Epoch: 22 cost time: 3.8137502670288086
# Epoch: 22, Steps: 30 | Train Loss: 0.4501283 Vali Loss: 0.9618362 Test Loss: 0.3968000
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.0001350851717672993
# Epoch: 23 cost time: 3.7844972610473633
# Epoch: 23, Steps: 30 | Train Loss: 0.4480381 Vali Loss: 0.9674724 Test Loss: 0.4027245
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.00012157665459056935
# Epoch: 24 cost time: 3.7443249225616455
# Epoch: 24, Steps: 30 | Train Loss: 0.4485399 Vali Loss: 0.9632542 Test Loss: 0.3982517
# EarlyStopping counter: 11 out of 100
# Updating learning rate to 0.00010941898913151242
# Epoch: 25 cost time: 4.535274982452393
# Epoch: 25, Steps: 30 | Train Loss: 0.4474640 Vali Loss: 0.9642033 Test Loss: 0.4016760
# EarlyStopping counter: 12 out of 100
# Updating learning rate to 9.847709021836118e-05
# Epoch: 26 cost time: 3.742678165435791
# Epoch: 26, Steps: 30 | Train Loss: 0.4466300 Vali Loss: 0.9680997 Test Loss: 0.3984744
# EarlyStopping counter: 13 out of 100
# Updating learning rate to 8.862938119652506e-05
# Epoch: 27 cost time: 3.8436288833618164
# Epoch: 27, Steps: 30 | Train Loss: 0.4457014 Vali Loss: 0.9657845 Test Loss: 0.4007165
# EarlyStopping counter: 14 out of 100
# Updating learning rate to 7.976644307687256e-05
# Epoch: 28 cost time: 3.7635304927825928
# Epoch: 28, Steps: 30 | Train Loss: 0.4442416 Vali Loss: 0.9695212 Test Loss: 0.4026839
# EarlyStopping counter: 15 out of 100
# Updating learning rate to 7.17897987691853e-05
# Epoch: 29 cost time: 3.86342453956604
# Epoch: 29, Steps: 30 | Train Loss: 0.4444906 Vali Loss: 0.9690772 Test Loss: 0.3989131
# EarlyStopping counter: 16 out of 100
# Updating learning rate to 6.461081889226677e-05
# Epoch: 30 cost time: 3.881075143814087
# Epoch: 30, Steps: 30 | Train Loss: 0.4439867 Vali Loss: 0.9684238 Test Loss: 0.3990996
# EarlyStopping counter: 17 out of 100
# Updating learning rate to 5.8149737003040094e-05
# Epoch: 31 cost time: 3.7658684253692627
# Epoch: 31, Steps: 30 | Train Loss: 0.4439677 Vali Loss: 0.9688971 Test Loss: 0.4036170
# EarlyStopping counter: 18 out of 100
# Updating learning rate to 5.233476330273609e-05
# Epoch: 32 cost time: 4.436225891113281
# Epoch: 32, Steps: 30 | Train Loss: 0.4447082 Vali Loss: 0.9669125 Test Loss: 0.4055596
# EarlyStopping counter: 19 out of 100
# Updating learning rate to 4.7101286972462485e-05
# Epoch: 33 cost time: 3.745298385620117
# Epoch: 33, Steps: 30 | Train Loss: 0.4444415 Vali Loss: 0.9736335 Test Loss: 0.3995712
# EarlyStopping counter: 20 out of 100
# Updating learning rate to 4.239115827521624e-05
# Epoch: 34 cost time: 3.967630386352539
# Epoch: 34, Steps: 30 | Train Loss: 0.4438488 Vali Loss: 0.9704022 Test Loss: 0.4023854
# EarlyStopping counter: 21 out of 100
# Updating learning rate to 3.8152042447694614e-05
# Epoch: 35 cost time: 3.7944862842559814
# Epoch: 35, Steps: 30 | Train Loss: 0.4440291 Vali Loss: 0.9711745 Test Loss: 0.4005891
# EarlyStopping counter: 22 out of 100
# Updating learning rate to 3.433683820292515e-05
# >>>>>>>testing : test_ITSMixer_ETTh1_ftM_sl512_pl336_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2545
# mse:0.3760424852371216, mae:0.412467360496521, rse:0.5908068418502808