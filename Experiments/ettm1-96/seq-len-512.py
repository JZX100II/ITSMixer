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
        self.drop = nn.Dropout(0.38)

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
# >>>>>>>start training : test_ITSMixer_ETTm1_ftM_sl512_pl96_ebtimeF_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
# train 33953
# val 11425
# test 11425
# 	iters: 100, epoch: 1 | loss: 0.8929801
# 	speed: 0.0627s/iter; left time: 200.8091s
# Epoch: 1 cost time: 7.148982763290405
# Epoch: 1, Steps: 132 | Train Loss: 1.0699457 Vali Loss: 0.6368926 Test Loss: 0.5385505
# Validation loss decreased (inf --> 0.636893).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 2 | loss: 0.3478130
# 	speed: 0.1008s/iter; left time: 309.3228s
# Epoch: 2 cost time: 6.4977006912231445
# Epoch: 2, Steps: 132 | Train Loss: 0.4388782 Vali Loss: 0.4119366 Test Loss: 0.3329456
# Validation loss decreased (0.636893 --> 0.411937).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 3 | loss: 0.3087612
# 	speed: 0.1083s/iter; left time: 318.1085s
# Epoch: 3 cost time: 6.658504962921143
# Epoch: 3, Steps: 132 | Train Loss: 0.3250961 Vali Loss: 0.4038753 Test Loss: 0.3249589
# Validation loss decreased (0.411937 --> 0.403875).  Saving model ...
# Updating learning rate to 0.001
# 	iters: 100, epoch: 4 | loss: 0.3259531
# 	speed: 0.1081s/iter; left time: 303.2972s
# Epoch: 4 cost time: 6.601462364196777
# Epoch: 4, Steps: 132 | Train Loss: 0.3169680 Vali Loss: 0.3964691 Test Loss: 0.3276699
# Validation loss decreased (0.403875 --> 0.396469).  Saving model ...
# Updating learning rate to 0.0009000000000000001
# 	iters: 100, epoch: 5 | loss: 0.3069511
# 	speed: 0.1082s/iter; left time: 289.3158s
# Epoch: 5 cost time: 7.171108245849609
# Epoch: 5, Steps: 132 | Train Loss: 0.3087402 Vali Loss: 0.3999288 Test Loss: 0.3291320
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0008100000000000001
# 	iters: 100, epoch: 6 | loss: 0.2909714
# 	speed: 0.1082s/iter; left time: 274.8930s
# Epoch: 6 cost time: 7.382061243057251
# Epoch: 6, Steps: 132 | Train Loss: 0.3038824 Vali Loss: 0.3909866 Test Loss: 0.3237064
# Validation loss decreased (0.396469 --> 0.390987).  Saving model ...
# Updating learning rate to 0.0007290000000000002
# 	iters: 100, epoch: 7 | loss: 0.3035646
# 	speed: 0.1069s/iter; left time: 257.6304s
# Epoch: 7 cost time: 7.147239923477173
# Epoch: 7, Steps: 132 | Train Loss: 0.2999366 Vali Loss: 0.3871947 Test Loss: 0.3224754
# Validation loss decreased (0.390987 --> 0.387195).  Saving model ...
# Updating learning rate to 0.0006561000000000001
# 	iters: 100, epoch: 8 | loss: 0.2818831
# 	speed: 0.1018s/iter; left time: 231.8240s
# Epoch: 8 cost time: 6.821743726730347
# Epoch: 8, Steps: 132 | Train Loss: 0.2969461 Vali Loss: 0.3883399 Test Loss: 0.3223257
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.00059049
# 	iters: 100, epoch: 9 | loss: 0.2892456
# 	speed: 0.1064s/iter; left time: 228.1399s
# Epoch: 9 cost time: 6.693857669830322
# Epoch: 9, Steps: 132 | Train Loss: 0.2938867 Vali Loss: 0.3853374 Test Loss: 0.3212366
# Validation loss decreased (0.387195 --> 0.385337).  Saving model ...
# Updating learning rate to 0.000531441
# 	iters: 100, epoch: 10 | loss: 0.2777690
# 	speed: 0.1089s/iter; left time: 219.1236s
# Epoch: 10 cost time: 6.573672294616699
# Epoch: 10, Steps: 132 | Train Loss: 0.2912248 Vali Loss: 0.3871799 Test Loss: 0.3197916
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0004782969000000001
# 	iters: 100, epoch: 11 | loss: 0.2825157
# 	speed: 0.1087s/iter; left time: 204.4273s
# Epoch: 11 cost time: 6.847473621368408
# Epoch: 11, Steps: 132 | Train Loss: 0.2887063 Vali Loss: 0.3851452 Test Loss: 0.3220332
# Validation loss decreased (0.385337 --> 0.385145).  Saving model ...
# Updating learning rate to 0.0004304672100000001
# 	iters: 100, epoch: 12 | loss: 0.2841901
# 	speed: 0.1065s/iter; left time: 186.2372s
# Epoch: 12 cost time: 7.138035297393799
# Epoch: 12, Steps: 132 | Train Loss: 0.2864380 Vali Loss: 0.3800300 Test Loss: 0.3191274
# Validation loss decreased (0.385145 --> 0.380030).  Saving model ...
# Updating learning rate to 0.0003874204890000001
# 	iters: 100, epoch: 13 | loss: 0.2757268
# 	speed: 0.1057s/iter; left time: 170.8457s
# Epoch: 13 cost time: 7.080477714538574
# Epoch: 13, Steps: 132 | Train Loss: 0.2836766 Vali Loss: 0.3804019 Test Loss: 0.3180595
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0003486784401000001
# 	iters: 100, epoch: 14 | loss: 0.2862582
# 	speed: 0.1024s/iter; left time: 152.0941s
# Epoch: 14 cost time: 6.947788953781128
# Epoch: 14, Steps: 132 | Train Loss: 0.2820173 Vali Loss: 0.3802004 Test Loss: 0.3184119
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.0003138105960900001
# 	iters: 100, epoch: 15 | loss: 0.2904930
# 	speed: 0.1042s/iter; left time: 140.9793s
# Epoch: 15 cost time: 6.690311431884766
# Epoch: 15, Steps: 132 | Train Loss: 0.2802400 Vali Loss: 0.3836236 Test Loss: 0.3213329
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0002824295364810001
# 	iters: 100, epoch: 16 | loss: 0.2791202
# 	speed: 0.1086s/iter; left time: 132.5643s
# Epoch: 16 cost time: 6.647519826889038
# Epoch: 16, Steps: 132 | Train Loss: 0.2784303 Vali Loss: 0.3824195 Test Loss: 0.3185836
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.0002541865828329001
# 	iters: 100, epoch: 17 | loss: 0.2772237
# 	speed: 0.1084s/iter; left time: 118.0508s
# Epoch: 17 cost time: 6.597281217575073
# Epoch: 17, Steps: 132 | Train Loss: 0.2767297 Vali Loss: 0.3782750 Test Loss: 0.3178562
# Validation loss decreased (0.380030 --> 0.378275).  Saving model ...
# Updating learning rate to 0.0002287679245496101
# 	iters: 100, epoch: 18 | loss: 0.2848020
# 	speed: 0.1102s/iter; left time: 105.4254s
# Epoch: 18 cost time: 7.004528045654297
# Epoch: 18, Steps: 132 | Train Loss: 0.2754625 Vali Loss: 0.3856990 Test Loss: 0.3218944
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.0002058911320946491
# 	iters: 100, epoch: 19 | loss: 0.2735465
# 	speed: 0.1060s/iter; left time: 87.4188s
# Epoch: 19 cost time: 7.110594272613525
# Epoch: 19, Steps: 132 | Train Loss: 0.2746322 Vali Loss: 0.3773354 Test Loss: 0.3156227
# Validation loss decreased (0.378275 --> 0.377335).  Saving model ...
# Updating learning rate to 0.00018530201888518417
# 	iters: 100, epoch: 20 | loss: 0.2796612
# 	speed: 0.1066s/iter; left time: 73.8802s
# Epoch: 20 cost time: 7.231295824050903
# Epoch: 20, Steps: 132 | Train Loss: 0.2731958 Vali Loss: 0.3778998 Test Loss: 0.3163258
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 0.00016677181699666576
# 	iters: 100, epoch: 21 | loss: 0.2728360
# 	speed: 0.1183s/iter; left time: 66.3884s
# Epoch: 21 cost time: 7.139798164367676
# Epoch: 21, Steps: 132 | Train Loss: 0.2716940 Vali Loss: 0.3808702 Test Loss: 0.3178860
# EarlyStopping counter: 2 out of 100
# Updating learning rate to 0.00015009463529699917
# 	iters: 100, epoch: 22 | loss: 0.2591391
# 	speed: 0.1017s/iter; left time: 43.6406s
# Epoch: 22 cost time: 6.785451412200928
# Epoch: 22, Steps: 132 | Train Loss: 0.2714828 Vali Loss: 0.3793228 Test Loss: 0.3167991
# EarlyStopping counter: 3 out of 100
# Updating learning rate to 0.0001350851717672993
# 	iters: 100, epoch: 23 | loss: 0.2682266
# 	speed: 0.1075s/iter; left time: 31.9140s
# Epoch: 23 cost time: 6.785049200057983
# Epoch: 23, Steps: 132 | Train Loss: 0.2704897 Vali Loss: 0.3815864 Test Loss: 0.3184115
# EarlyStopping counter: 4 out of 100
# Updating learning rate to 0.00012157665459056935
# 	iters: 100, epoch: 24 | loss: 0.2807079
# 	speed: 0.1094s/iter; left time: 18.0451s
# Epoch: 24 cost time: 6.648091077804565
# Epoch: 24, Steps: 132 | Train Loss: 0.2696821 Vali Loss: 0.3760197 Test Loss: 0.3164256
# Validation loss decreased (0.377335 --> 0.376020).  Saving model ...
# Updating learning rate to 0.00010941898913151242
# 	iters: 100, epoch: 25 | loss: 0.2743522
# 	speed: 0.1093s/iter; left time: 3.6070s
# Epoch: 25 cost time: 6.84669041633606
# Epoch: 25, Steps: 132 | Train Loss: 0.2688709 Vali Loss: 0.3808224 Test Loss: 0.3182462
# EarlyStopping counter: 1 out of 100
# Updating learning rate to 9.847709021836118e-05
# >>>>>>>testing : test_ITSMixer_ETTm1_ftM_sl512_pl96_ebtimeF_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 11425
# mse:0.29068055748939514, mae:0.3421705365180969, rse:0.5119670629501343