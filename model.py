import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from torchvision import transforms
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        
        self.n_RDBs_fomer =2
        self.n_ESTMs = 3
        self.n_RDBs_inter = 2
        
        self.CondNet = nn.Sequential(
            nn.Conv2d(11, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 64, 1))
        
        
        self.init_feature = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        
        self.deep_features = nn.ModuleList([RDG(G0=64, C=4, G=24, n_RDB=self.n_RDBs_fomer)])
        self.deep_features.extend([RDG(G0=64, C=4, G=24, n_RDB=self.n_RDBs_inter) for i in range(self.n_ESTMs)])
        
        self.ESTMs = nn.ModuleList([ESTM(64, self.n_RDBs_fomer)])
        self.ESTMs.extend([ESTM(64, self.n_RDBs_inter) for i in range(self.n_ESTMs)])
        
        self.Convs = nn.ModuleList([nn.Conv2d(64*self.n_RDBs_fomer, 64, kernel_size=1, stride=1, padding=0, bias=True)])
        self.Convs.extend([nn.Conv2d(64*self.n_RDBs_inter, 64, kernel_size=1, stride=1, padding=0, bias=True) for i in range(self.n_ESTMs)])
        
        self.Fusions = nn.ModuleList([Fusion() for i in range(self.n_ESTMs+1)])
        
        self.fusion_final = nn.Sequential(
            nn.Conv2d(64*(self.n_ESTMs+1), 64, 1, 1, 0, bias=False),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False))
                
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=True))

    def forward(self, x_left, x_right, edge_left_f, edge_right_f):
        edge_left_f = torch.cat(edge_left_f, dim=1)
        edge_right_f = torch.cat(edge_right_f, dim=1)
        edge_cond_left = self.CondNet(edge_left_f)   # the size of the conditional edge prior[b,w,h,64]
        edge_cond_right = self.CondNet(edge_right_f)

        x_left_upscale = F.interpolate(x_left, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        init_buffer_left = buffer_left
        init_buffer_right = buffer_right
        temp_left = []
        temp_right = []
        for i in range(self.n_ESTMs+1):
            catfea_left = self.deep_features[i](buffer_left)
            catfea_right = self.deep_features[i](buffer_right)

            buffer_leftT, buffer_rightT \
                = self.ESTMs[i](catfea_left, catfea_right, edge_cond_left, edge_cond_right)

            buffer_left = self.Convs[i](catfea_left)
            buffer_right = self.Convs[i](catfea_right)
            
            buffer_left = self.Fusions[i](torch.cat([buffer_left, buffer_leftT], dim=1))
            buffer_right = self.Fusions[i](torch.cat([buffer_right, buffer_rightT], dim=1))
            temp_left.append(buffer_left)
            temp_right.append(buffer_right)
        # fuse all the stereo feature with edge features
        buffer_left = self.fusion_final(torch.cat(temp_left, dim=1))
        buffer_right = self.fusion_final(torch.cat(temp_right, dim=1))
        
        out_left = self.upscale(buffer_left+init_buffer_left) + x_left_upscale
        out_right = self.upscale(buffer_right+init_buffer_right) + x_right_upscale
        del buffer_leftT, buffer_rightT, temp_left, temp_right, init_buffer_right, init_buffer_left, \
            edge_left_f, edge_right_f, edge_cond_left, edge_cond_right, x_left_upscale, x_right_upscale
        
        return out_left, out_right

class SFTLayer(nn.Module): #Edge-Adaptive Spatial Feature Transform (EASFT)
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(64, 64, 1)
        self.SFT_scale_conv1 = nn.Conv2d(64, 128, 1)
        self.SFT_shift_conv0 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv1 = nn.Conv2d(64, 128, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        del temp
        return buffer_cat


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True),
            RDB(G0=64, C=4, G=32),
            CALayer(64))

    def forward(self, x):
        x = self.fusion(x)
        return x


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


            
class ESTM(nn.Module): ## Edge-guided stereo attention mechanism (ESAM)
    def __init__(self, channels, n_RDBs):
        super(ESTM, self).__init__()
        self.bv = nn.Conv2d(n_RDBs*channels, channels, 1, 1, 0, bias=True)
        self.bq = nn.Conv2d((n_RDBs)*channels, channels, 1, 1, 0, groups=n_RDBs, bias=True)
        self.bs = nn.Conv2d((n_RDBs)*channels, channels, 1, 1, 0, groups=n_RDBs, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(n_RDBs * channels)
        self.sft = SFTLayer()

    def __call__(self, catfea_left, catfea_right, edge_cond_left, edge_cond_right):
        x_left = self.bv(catfea_left)
        x_right = self.bv(catfea_right)
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.sft([catfea_left, edge_cond_left])))
        b, c, h, w = Q.shape
        K = self.bs(self.rb(self.sft([catfea_right, edge_cond_right])))
        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left + x_leftT
        out_right = x_right + x_rightT
        
        del x_leftT, x_rightT, M_right_to_left, M_left_to_right

        return out_left, out_right
            

if __name__ == "__main__":
    net = Net(upscale_factor=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))