# -*- coding: utf-8 -*-
import torch as pt

class DoubleConv(pt.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),  
            pt.nn.LeakyReLU(inplace=True),
            pt.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),
            )

        self.residual_upsampler = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            pt.nn.InstanceNorm3d(out_ch))

        self.relu=pt.nn.LeakyReLU(inplace=True)

    def forward(self, input):
        return self.relu(self.conv(input)+self.residual_upsampler(input))


class Deconv3D_Block(pt.nn.Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv3D_Block, self).__init__()
        
        self.deconv = pt.nn.Sequential(
                        pt.nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel,kernel,kernel), 
                                    stride=(stride,stride,stride), padding=(padding, padding, padding), output_padding=0, bias=True),
                        pt.nn.LeakyReLU())
    
    def forward(self, x):
        
        return self.deconv(x)

class FinetuneNet(pt.nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super(FinetuneNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = pt.nn.MaxPool3d((2,2,2))
        self.conv2 = DoubleConv(32, 32)
        self.pool2 = pt.nn.MaxPool3d((2,2,2)) 
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = pt.nn.MaxPool3d((2,2,2))
        self.conv4 = DoubleConv(64, 128)
        self.pool4 = pt.nn.MaxPool3d((2,2,2))
        self.conv5 = DoubleConv(128, 256)
        self.up6_seg = Deconv3D_Block(256, 128, 4, stride=2)
        self.conv6_seg = DoubleConv(256, 128)
        self.up7_seg = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_seg = DoubleConv(128, 64)
        self.up8_seg = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_seg = DoubleConv(64, 32)
        self.up9_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_seg = DoubleConv(64, 32)
        self.conv10 = pt.nn.Conv3d(32, out_channels, 1)

        # self.pointwise = pt.nn.Sequential(
        #     pt.nn.Conv3d(32,32,1),
        #     pt.nn.InstanceNorm3d(32),  
        #     pt.nn.LeakyReLU(inplace=True)
        # )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6_seg = self.up6_seg(c5)
        merge6_seg = pt.cat([up_6_seg, c4], dim=1)
        c6_seg = self.conv6_seg(merge6_seg)
        up_7_seg = self.up7_seg(c6_seg)
        merge7_seg = pt.cat([up_7_seg, c3], dim=1)
        c7_seg = self.conv7_seg(merge7_seg)
        up_8_seg = self.up8_seg(c7_seg)
        merge8_seg = pt.cat([up_8_seg, c2], dim=1)
        c8_seg = self.conv8_seg(merge8_seg)
        up_9_seg = self.up9_seg(c8_seg)
        merge9_seg = pt.cat([up_9_seg, c1], dim=1)
        c9_seg = self.conv9_seg(merge9_seg)
        c10_seg = self.conv10(c9_seg)
        out_seg = pt.nn.Sigmoid()(c10_seg)

        return out_seg