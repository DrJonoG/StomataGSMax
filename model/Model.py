import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class Upsample(nn.Module):
    def __init__(self,ch_in):
        super(Upsample,self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(ch_in, ch_in // 2,kernel_size=2,stride=2,padding=0,bias=True),
            #nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Input(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Input,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class BranchConvBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(BranchConvBlock,self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

        self.m1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x = torch.cat((x1,x2), 1)
        #x = self.m1(x)

        return x

class BranchConvBlockIncrementalUp(nn.Module):
    def __init__(self,depth,ch_in, ch_out):
        super(BranchConvBlockIncrementalUp,self).__init__()

        self.depth = depth
        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ch_out = int(ch_out / depth)

        self.b0 =  nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out)
        )

        self.b1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.b3 =  nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.m1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(ch_in // 2, ch_in // 2,kernel_size=2,stride=2,padding=0,bias=True),
            #nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

        self.Dropout = nn.Dropout(0.2)

    def forward(self,x):
        if self.depth == 1:
            x = self.b0(x)
        elif self.depth == 2:
            x0 = self.b0(x)
            x1 = self.b1(x)
            x = torch.cat((x0,x1), 1)
        else:
            x0 = self.b0(x)
            x1 = self.b1(x)
            x2 = self.b2(x)
            x3 = self.b3(x)
            x = torch.cat((x0,x1,x2,x3), 1)
            #x = self.Dropout(x)
        x = self.up(x)
        return x

class BranchConvBlockIncremental(nn.Module):
    def __init__(self,depth,ch_in, ch_out):
        super(BranchConvBlockIncremental,self).__init__()

        self.depth = depth
        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ch_out = int(ch_out / depth)

        self.b0 =  nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.b1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.b3 =  nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )


    def forward(self,x):
        x = self.Maxpool(x)
        if self.depth == 1:
            x = self.b0(x)
        elif self.depth == 2:
            x0 = self.b0(x)
            x1 = self.b1(x)
            x = torch.cat((x0,x1), 1)
        else:
            x0 = self.b0(x)
            x1 = self.b1(x)
            x2 = self.b2(x)
            x3 = self.b3(x)
            x = torch.cat((x0,x1,x2,x3), 1)
        return x

class BranchConvBlockUp(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(BranchConvBlockUp,self).__init__()

        self.m1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.m1(x)
        return x

class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()

        #self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Dropout = nn.Dropout(0.2)

        self.Conv1 = Input(ch_in=img_ch,ch_out=32)
        self.Conv2 = BranchConvBlockIncremental(1,ch_in=32,ch_out=64)
        self.Conv3 = BranchConvBlockIncremental(2,ch_in=64,ch_out=128)
        self.Conv4 = BranchConvBlockIncremental(2,ch_in=128,ch_out=256)
        self.Conv5 = BranchConvBlockIncremental(4,ch_in=256,ch_out=512)
        self.Conv6 = BranchConvBlockIncremental(4,ch_in=512,ch_out=1024)


        self.Up6 = BranchConvBlockIncrementalUp(4, ch_in=1024, ch_out=512)
        self.Att6 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv6 = BranchConvBlockUp(ch_in=1024, ch_out=512)

        self.Up5 = BranchConvBlockIncrementalUp(4, ch_in=512, ch_out=256)
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = BranchConvBlockUp(ch_in=512, ch_out=256)

        self.Up4 = BranchConvBlockIncrementalUp(2, ch_in=256, ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = BranchConvBlockUp(ch_in=256, ch_out=128)

        self.Up3 = BranchConvBlockIncrementalUp(2, ch_in=128, ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = BranchConvBlockUp(ch_in=128, ch_out=64)

        self.Up2 = BranchConvBlockIncrementalUp(1, ch_in=64, ch_out=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = BranchConvBlockUp(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0),
            #nn.ReLU(inplace=True)
        )

    def forward(self,x):
        # encoding path

        x1 = self.Conv1(x)

        x2 = self.Conv2(x1)

        x3 = self.Conv3(x2)

        x4 = self.Conv4(x3)

        x5 = self.Conv5(x4)

        x6 = self.Conv6(x5)

        d6 = self.Up6(x6)
        x5 = self.Att6(d6,x5)
        d6 = torch.cat((x5,d6),dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(x5)
        x4 = self.Att5(d5,x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4,x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3,x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2,x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
