import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import GraphReasoning
from thop import profile

class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Encoder_ir(nn.Module):
    def __init__(self, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Encoder_ir, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        # channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 32, bias=False)
        self.fc2 = nn.Linear(32, 512, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x1 = self.maxpool(x1)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x2 = self.maxpool(x2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)
        x3_max = self.maxpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3_avg = self.avgpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3 = torch.mul(x3_max, x3_avg)

        # channel attention
        x_gap = self.gap(x3)
        x_gap = x_gap.view(x_gap.size(0), -1)
        x_gap = self.relu(self.fc1(x_gap))
        x_gap = self.fc2(x_gap)
        x_weights = self.sigmoid(x_gap)
        x_weights = x_weights.view(x3.size(0), x3.size(1), 1, 1)
        x3 = x3 * x_weights.expand_as(x3)

        return x3, x2, x1

class Encoder_vis(nn.Module):
    def __init__(self, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Encoder_vis, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        # channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 32, bias=False)
        self.fc2 = nn.Linear(32, 512, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x1 = self.maxpool(x1)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x2 = self.maxpool(x2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)
        x3_max = self.maxpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3_avg = self.avgpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3 = torch.add(x3_max, x3_avg)

        # channel attention
        x_gap = self.gap(x3)
        x_gap = x_gap.view(x_gap.size(0), -1)
        x_gap = self.relu(self.fc1(x_gap))
        x_gap = self.fc2(x_gap)
        x_weights = self.sigmoid(x_gap)
        x_weights = x_weights.view(x3.size(0), x3.size(1), 1, 1)
        x3 = x3 * x_weights.expand_as(x3)

        return x3, x2, x1

class CGR(nn.Module):
    def __init__(self, n_class=2, n_iter=2, chnn_side=(512, 256, 128), chnn_targ=(512, 128, 32, 4), rd_sc=32, dila=(4, 8, 16)):
        super().__init__()
        self.n_graph = len(chnn_side)
        n_node = len(dila)
        graph = [GraphReasoning(ii, rd_sc, dila, n_iter) for ii in chnn_side]
        self.graph = nn.ModuleList(graph)
        C_cat = [nn.Sequential(
            nn.Conv2d(ii//rd_sc*n_node, ii//rd_sc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ii//rd_sc),
            nn.ReLU(inplace=True))
            for ii in (chnn_side+chnn_side)]
        self.C_cat = nn.ModuleList(C_cat)
        idx = [ii for ii in range(len(chnn_side))]
        C_up = [nn.Sequential(
            nn.Conv2d(chnn_targ[ii]+chnn_side[ii]//rd_sc, chnn_targ[ii+1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chnn_targ[ii+1]),
            nn.ReLU(inplace=True))
            for ii in (idx+idx)]
        self.C_up = nn.ModuleList(C_up)
        self.C_cls = nn.Conv2d(chnn_targ[-1]*2, 1, 1)  # n_class -> 1

    def forward(self, inputs):
        img, depth = inputs
        cas_rgb, cas_dep = img[0], depth[0]
        nd_rgb, nd_dep, nd_key = None, None, False
        for ii in range(self.n_graph):
            feat_rgb, feat_dep = self.graph[ii]([img[ii], depth[ii], nd_rgb, nd_dep], nd_key)
            feat_rgb = torch.cat(feat_rgb, 1)
            feat_rgb = self.C_cat[ii](feat_rgb)
            feat_dep = torch.cat(feat_dep, 1)
            feat_dep = self.C_cat[self.n_graph+ii](feat_dep)
            nd_rgb, nd_dep, nd_key = feat_rgb, feat_dep, True
            cas_rgb = torch.cat((feat_rgb, cas_rgb), 1)
            cas_rgb = F.interpolate(cas_rgb, scale_factor=2, mode='bilinear', align_corners=True)
            cas_rgb = self.C_up[ii](cas_rgb)
            cas_dep = torch.cat((feat_dep, cas_dep), 1)
            cas_dep = F.interpolate(cas_dep, scale_factor=2, mode='bilinear', align_corners=True)
            cas_dep = self.C_up[self.n_graph+ii](cas_dep)
        feat = torch.cat((cas_rgb, cas_dep), dim=1)
        out = self.C_cls(feat)
        return out

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.vis_conv = Encoder_vis()

        self.inf_conv = Encoder_ir()

        self.graph = CGR()

    def forward(self, image_vis, image_ir):
        image_vis = image_vis[:, : 1]
        image_ir = image_ir

        image_vis = self.vis_conv(image_vis)

        image_ir = self.inf_conv(image_ir)

        x = self.graph([image_ir, image_vis])

        return x




