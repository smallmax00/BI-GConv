import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s
from .edge_aware_gcn import GRU_EAGCN



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class EDGModule(nn.Module):
    def __init__(self, channel):
        super(EDGModule, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2*channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)

        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3):  # 16x16, 32x32, 64x64
        up_x1 = self.conv_upsample1(self.upsample(x1))
        conv_x2 = self.conv_1(x2)
        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))

        up_x2 = self.conv_upsample2(self.upsample(x2))
        conv_x3 = self.conv_2(x3)
        cat_x3 = self.conv_concat3(torch.cat((up_x2, conv_x3), 1))

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2))
        conv_cat_x3 = self.conv_3(cat_x3)
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, conv_cat_x3), 1))
        x = self.conv5(cat_x4)
        return x


class SEG_Module(nn.Module):

    def __init__(self, channel):
        super(SEG_Module, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)



        self.conv5 = nn.Conv2d(4 * channel, 2, 1)
        self.GCN = GRU_EAGCN(num_in=2, plane_mid=1, mids=32)

    def forward(self, x1, x2, x3, edge):  
        up_x1 = self.conv_upsample1(self.upsample(x1))
        conv_x2 = self.conv_1(x2)
        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))

        up_x2 = self.conv_upsample2(self.upsample(x2))
        conv_x3 = self.conv_2(x3)
        cat_x3 = self.conv_concat3(torch.cat((up_x2, conv_x3), 1))

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2))
        conv_cat_x3 = self.conv_3(cat_x3)
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, conv_cat_x3), 1))
        seg = self.conv5(cat_x4)

        seg_gcn = self.GCN(seg, edge) 
        seg_gcn = seg_gcn + seg

        seg = self.upsample(seg_gcn)


        return seg


class ODOC_seg_edge(nn.Module):
    
    def __init__(self, channel=64):
        super(ODOC_seg_edge, self).__init__()
       
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)

        self.rfb2_1 = BasicConv2d(256, channel, 1)
        self.rfb3_1 = BasicConv2d(512, channel, 1)
        self.rfb4_1 = BasicConv2d(1024, channel, 1)
        self.rfb5_1 = BasicConv2d(2048, channel, 1)

        
        self.edge = EDGModule(channel)
        
        self.seg_layer = SEG_Module(channel)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      

        
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)     
        x3 = self.resnet.layer3(x2)     
        x4 = self.resnet.layer4(x3)     


        x1_rfb = self.rfb2_1(x1)
        x2_rfb = self.rfb3_1(x2)        
        x3_rfb = self.rfb4_1(x3)        
        x4_rfb = self.rfb5_1(x4)        

        edge_feat = self.edge(x3_rfb, x2_rfb, x1_rfb)  

        edge = F.interpolate(edge_feat, size=(32, 32), mode='bilinear', align_corners=True)

        seg = self.seg_layer(x4_rfb, x3_rfb, x2_rfb, edge)  

        seg_output = torch.sigmoid(seg)


        return seg_output, torch.sigmoid(edge_feat) 


