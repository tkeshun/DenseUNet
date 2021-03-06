import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

acti_enc = torch.nn.LeakyReLU(0.2)
acti_dec = torch.nn.ReLU()
acti_out = torch.nn.Sigmoid()
class model(nn.Module):  
    def __init__(self):
        super(model,self).__init__()
        #1 -> 16
        self.down1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(5,5),stride=(2,2),padding=(2,2))
        self.bn1 = nn.BatchNorm2d(16)
        #16 -> 32
        self.down2   = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(5,5),stride=(2,2),padding=(2,2))
        self.bn2     = nn.BatchNorm2d(32)
        #32 -> 64
        self.down3   = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,5),stride=(2,2),padding=(2,2))
        self.bn3     = nn.BatchNorm2d(64)
        #64 -> 128
        self.down4   = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,5),stride=(2,2),padding=(2,2))
        self.bn4     = nn.BatchNorm2d(128)
        #128 -> 256
        self.down5   = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=(2,2),padding=(2,2))
        self.bn5     = nn.BatchNorm2d(256)
        #256 -> 512
        self.down6   = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=(2,2),padding=(2,2))
        self.bn6     = nn.BatchNorm2d(512)

        #512 -> 256
        self.up1     = nn.ConvTranspose2d(512,256,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        self.bn7     = nn.BatchNorm2d(256)
        self.dout1   = nn.Dropout(p=0.5)

        #256 -> 128
        self.up2     = nn.ConvTranspose2d(512,128,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        self.bn8     = nn.BatchNorm2d(128)
        self.dout2   = nn.Dropout(p=0.5)
        #128 -> 64
        self.up3     = nn.ConvTranspose2d(256,64,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        self.bn9     = nn.BatchNorm2d(64)
        self.dout3   = nn.Dropout(p=0.5)
        #64 -> 32
        self.up4     = nn.ConvTranspose2d(128,32,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        self.bn10    = nn.BatchNorm2d(32)

        #32 -> 16
        self.up5     = nn.ConvTranspose2d(64,16,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        self.bn11    = nn.BatchNorm2d(16)
        #16 -> 1
        
        self.up6   = nn.ConvTranspose2d(32,1,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        
    def forward(self,x):
        #encoder
        layer1  = acti_enc(self.bn1(self.down1(x)))
        layer2  = acti_enc(self.bn2(self.down2(layer1)))
        layer3  = acti_enc(self.bn3(self.down3(layer2)))
        layer4  = acti_enc(self.bn4(self.down4(layer3)))
        layer5  = acti_enc(self.bn5(self.down5(layer4)))
        layer6  = acti_enc(self.bn6(self.down6(layer5)))
        
        #decoder
        dlayer1 = acti_dec(self.dout1(self.bn7(self.up1(layer6))))
        dlayer2 = acti_dec(self.dout2(self.bn8(self.up2(torch.cat([dlayer1,layer5],dim=1)))))
        dlayer3 = acti_dec(self.dout3(self.bn9(self.up3(torch.cat([dlayer2,layer4],dim=1)))))
        dlayer4 = acti_dec(self.bn10(self.up4(torch.cat([dlayer3,layer3],dim=1))))
        dlayer5 = acti_dec(self.bn11(self.up5(torch.cat([dlayer4,layer2],dim=1))))
        Mask = acti_out(self.up6(torch.cat([dlayer5,layer1],dim=1)))
        output  = torch.multiply(Mask,x)
        return output 