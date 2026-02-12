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
        self.drop = nn.Dropout(0.4)

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
# >>>>>>>start training : test_ITSMixer_ETTh1_ftM_sl512_pl192_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 7937
# val 2689
# test 2689
# Epoch: 1 cost time: 1.9723448753356934
# Epoch: 1, Steps: 31 | Train Loss: 0.8378697 Vali Loss: 1.2456198 Test Loss: 0.6860642
# Validation loss decreased (inf --> 1.245620).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 2 cost time: 1.688866376876831
# Epoch: 2, Steps: 31 | Train Loss: 0.6434727 Vali Loss: 0.8641227 Test Loss: 0.4829523
# Validation loss decreased (1.245620 --> 0.864123).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 3 cost time: 1.6692686080932617
# Epoch: 3, Steps: 31 | Train Loss: 0.5135156 Vali Loss: 0.8215361 Test Loss: 0.4252397
# Validation loss decreased (0.864123 --> 0.821536).  Saving model ...
# Updating learning rate to 0.001
# Epoch: 4 cost time: 2.271106004714966
# Epoch: 4, Steps: 31 | Train Loss: 0.4621998 Vali Loss: 0.8193698 Test Loss: 0.4095207
# Validation loss decreased (0.821536 --> 0.819370).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# Epoch: 5 cost time: 1.6859705448150635
# Epoch: 5, Steps: 31 | Train Loss: 0.4418819 Vali Loss: 0.8108377 Test Loss: 0.4017686
# Validation loss decreased (0.819370 --> 0.810838).  Saving model ...
# Updating learning rate to 0.0008100000000000001
# Epoch: 6 cost time: 1.6630640029907227
# Epoch: 6, Steps: 31 | Train Loss: 0.4313223 Vali Loss: 0.8125685 Test Loss: 0.3994474
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0007290000000000002
# Epoch: 7 cost time: 1.6673812866210938
# Epoch: 7, Steps: 31 | Train Loss: 0.4251723 Vali Loss: 0.8102196 Test Loss: 0.3986669
# Validation loss decreased (0.810838 --> 0.810220).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# Epoch: 8 cost time: 2.241739273071289
# Epoch: 8, Steps: 31 | Train Loss: 0.4202579 Vali Loss: 0.8149003 Test Loss: 0.3977889
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.00059049
# Epoch: 9 cost time: 1.651841402053833
# Epoch: 9, Steps: 31 | Train Loss: 0.4161579 Vali Loss: 0.8045080 Test Loss: 0.3962480
# Validation loss decreased (0.810220 --> 0.804508).  Saving model ...
# Updating learning rate to 0.000531441
# Epoch: 10 cost time: 1.6495709419250488
# Epoch: 10, Steps: 31 | Train Loss: 0.4134856 Vali Loss: 0.8077774 Test Loss: 0.3960693
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004782969000000001
# Epoch: 11 cost time: 1.6599266529083252
# Epoch: 11, Steps: 31 | Train Loss: 0.4116932 Vali Loss: 0.7995688 Test Loss: 0.3956570
# Validation loss decreased (0.804508 --> 0.799569).  Saving model ...
# Updating learning rate to 0.0004304672100000001
# Epoch: 12 cost time: 1.9834282398223877
# Epoch: 12, Steps: 31 | Train Loss: 0.4093789 Vali Loss: 0.8102533 Test Loss: 0.3983073
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003874204890000001
# Epoch: 13 cost time: 1.6555371284484863
# Epoch: 13, Steps: 31 | Train Loss: 0.4069662 Vali Loss: 0.8089352 Test Loss: 0.3970767
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003486784401000001
# Epoch: 14 cost time: 1.6639103889465332
# Epoch: 14, Steps: 31 | Train Loss: 0.4052876 Vali Loss: 0.8099539 Test Loss: 0.3976819
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0003138105960900001
# Epoch: 15 cost time: 1.6982688903808594
# Epoch: 15, Steps: 31 | Train Loss: 0.4032140 Vali Loss: 0.8144272 Test Loss: 0.3990845
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0002824295364810001
# Epoch: 16 cost time: 1.8502376079559326
# Epoch: 16, Steps: 31 | Train Loss: 0.4013101 Vali Loss: 0.8165429 Test Loss: 0.3989736
# EarlyStopping counter: 5 out of 100
# Updating learning rate to 0.0002541865828329001
# Epoch: 17 cost time: 1.6763203144073486
# Epoch: 17, Steps: 31 | Train Loss: 0.4004348 Vali Loss: 0.8235995 Test Loss: 0.3989165
# EarlyStopping counter: 6 out of 100
# Updating learning rate to 0.0002287679245496101
# Epoch: 18 cost time: 1.6791465282440186
# Epoch: 18, Steps: 31 | Train Loss: 0.3990179 Vali Loss: 0.8257270 Test Loss: 0.3990153
# EarlyStopping counter: 7 out of 100
# Updating learning rate to 0.0002058911320946491
# Epoch: 19 cost time: 1.664515733718872
# Epoch: 19, Steps: 31 | Train Loss: 0.3982861 Vali Loss: 0.8286911 Test Loss: 0.3977928
# EarlyStopping counter: 8 out of 100
# Updating learning rate to 0.00018530201888518417
# Epoch: 20 cost time: 1.8737142086029053
# Epoch: 20, Steps: 31 | Train Loss: 0.3967254 Vali Loss: 0.8173674 Test Loss: 0.4024214
# EarlyStopping counter: 9 out of 100
# Updating learning rate to 0.00016677181699666576
# Epoch: 21 cost time: 1.7337989807128906
# Epoch: 21, Steps: 31 | Train Loss: 0.3970989 Vali Loss: 0.8202410 Test Loss: 0.3999325
# EarlyStopping counter: 10 out of 100
# Updating learning rate to 0.00015009463529699917
# Epoch: 22 cost time: 1.6784183979034424
# Epoch: 22, Steps: 31 | Train Loss: 0.3960474 Vali Loss: 0.8220720 Test Loss: 0.3997478
# EarlyStopping counter: 11 out of 100
# Updating learning rate to 0.0001350851717672993
# Epoch: 23 cost time: 1.6491725444793701
# Epoch: 23, Steps: 31 | Train Loss: 0.3950239 Vali Loss: 0.8248088 Test Loss: 0.4003263
# EarlyStopping counter: 12 out of 100
# Updating learning rate to 0.00012157665459056935
# Epoch: 24 cost time: 2.2869131565093994
# Epoch: 24, Steps: 31 | Train Loss: 0.3941690 Vali Loss: 0.8337704 Test Loss: 0.4022730
# EarlyStopping counter: 13 out of 100
# Updating learning rate to 0.00010941898913151242
# Epoch: 25 cost time: 1.6883783340454102
# Epoch: 25, Steps: 31 | Train Loss: 0.3934590 Vali Loss: 0.8342999 Test Loss: 0.4006434
# EarlyStopping counter: 14 out of 100
# Updating learning rate to 9.847709021836118e-05
# Epoch: 26 cost time: 1.8382625579833984
# Epoch: 26, Steps: 31 | Train Loss: 0.3931984 Vali Loss: 0.8381830 Test Loss: 0.4022493
# EarlyStopping counter: 15 out of 100
# Updating learning rate to 8.862938119652506e-05
# Epoch: 27 cost time: 1.8204030990600586
# Epoch: 27, Steps: 31 | Train Loss: 0.3926658 Vali Loss: 0.8352925 Test Loss: 0.4002019
# EarlyStopping counter: 16 out of 100
# Updating learning rate to 7.976644307687256e-05
# Epoch: 28 cost time: 2.248159408569336
# Epoch: 28, Steps: 31 | Train Loss: 0.3923735 Vali Loss: 0.8322021 Test Loss: 0.4011087
# EarlyStopping counter: 17 out of 100
# Updating learning rate to 7.17897987691853e-05
# Epoch: 29 cost time: 1.6693074703216553
# Epoch: 29, Steps: 31 | Train Loss: 0.3919951 Vali Loss: 0.8376439 Test Loss: 0.4013937
# EarlyStopping counter: 18 out of 100
# Updating learning rate to 6.461081889226677e-05
# Epoch: 30 cost time: 1.6755635738372803
# Epoch: 30, Steps: 31 | Train Loss: 0.3917559 Vali Loss: 0.8319454 Test Loss: 0.4019060
# EarlyStopping counter: 19 out of 100
# Updating learning rate to 5.8149737003040094e-05
# Epoch: 31 cost time: 1.692767858505249
# Epoch: 31, Steps: 31 | Train Loss: 0.3922000 Vali Loss: 0.8341247 Test Loss: 0.4008163
# EarlyStopping counter: 20 out of 100
# Updating learning rate to 5.233476330273609e-05
# Epoch: 32 cost time: 2.4054110050201416
# Epoch: 32, Steps: 31 | Train Loss: 0.3916643 Vali Loss: 0.8308026 Test Loss: 0.4008720
# EarlyStopping counter: 21 out of 100
# Updating learning rate to 4.7101286972462485e-05
# Epoch: 33 cost time: 1.6792080402374268
# Epoch: 33, Steps: 31 | Train Loss: 0.3914394 Vali Loss: 0.8295833 Test Loss: 0.4012119
# EarlyStopping counter: 22 out of 100
# Updating learning rate to 4.239115827521624e-05
# Epoch: 34 cost time: 1.6725659370422363
# Epoch: 34, Steps: 31 | Train Loss: 0.3906573 Vali Loss: 0.8359557 Test Loss: 0.4004348
# EarlyStopping counter: 23 out of 100
# Updating learning rate to 3.8152042447694614e-05
# Epoch: 35 cost time: 1.6851613521575928
# Epoch: 35, Steps: 31 | Train Loss: 0.3908073 Vali Loss: 0.8343714 Test Loss: 0.4008755
# EarlyStopping counter: 24 out of 100
# Updating learning rate to 3.433683820292515e-05
# >>>>>>>testing : test_ITSMixer_ETTh1_ftM_sl512_pl192_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 2689
# mse:0.3832440972328186, mae:0.40807029604911804, rse:0.5888321399688721