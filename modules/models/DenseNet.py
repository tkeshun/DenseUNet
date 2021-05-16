import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Dense blockの形式を画像のDenseNetを参考にし、
→BN,relu,conv,→にした
"""
#Dense UNet
class model(nn.Module):  
    def __init__(self):

        super(model,self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(1, 1),
                stride=1,

            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        #Dense block & DownSampling
        ###dense1 1 -> 32### 
        self.dense1_c1  = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d1_bn1     = nn.BatchNorm2d(32)
        self.dense1_c2  = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d1_bn2     = nn.BatchNorm2d(64)
        self.dense1_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d1_bn3     = nn.BatchNorm2d(96)

        #down sampling
        self.DS1 = nn.Sequential(
             #nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2),
        ) 
        
      
        ###dense2 32 -> 32
        self.dense2_c1  = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d2_bn1     = nn.BatchNorm2d(32)
        self.dense2_c2  = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d2_bn2     = nn.BatchNorm2d(64)
        self.dense2_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d2_bn3     = nn.BatchNorm2d(96)

        #down sampling
        self.DS2 = nn.Sequential(
             #nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2),
        )

        ###dense3 32 -> 32
        self.dense3_c1  = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d3_bn1     = nn.BatchNorm2d(32)
        self.dense3_c2  = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d3_bn2     = nn.BatchNorm2d(64)
        self.dense3_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d3_bn3     = nn.BatchNorm2d(96)

        #down sampling
        self.DS3 = nn.Sequential(
             #nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2),
        ) 
        
        ###dense4 32 -> 32
        self.dense4_c1  = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))#concat
        self.d4_bn1     = nn.BatchNorm2d(32)
        self.dense4_c2  = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d4_bn2     = nn.BatchNorm2d(64)
        self.dense4_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d4_bn3     = nn.BatchNorm2d(96)

        #down sampling
        self.DS4 = nn.Sequential(
             #nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2),
        )

        ###dense5 32 -> 32
        self.dense5_c1  = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))#concat
        self.d5_bn1     = nn.BatchNorm2d(32)
        self.dense5_c2  = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d5_bn2     = nn.BatchNorm2d(64)
        self.dense5_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d5_bn3     = nn.BatchNorm2d(96)
        
        #up sampling
        self.UP1 = nn.Sequential(
             nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2)
        ) 
        ###dense6 32 -> 32
        self.dense6_c1  = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))#concat
        self.d6_bn1     = nn.BatchNorm2d(64)
        self.dense6_c2  = nn.Conv2d(in_channels=64+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d6_bn2     = nn.BatchNorm2d(96)
        self.dense6_c3  = nn.Conv2d(in_channels=96+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d6_bn3     = nn.BatchNorm2d(128)

        #up sampling
        self.UP2 = nn.Sequential(
             nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2)
        ) 

        ###dense7 32 -> 32
        self.dense7_c1  = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))#concat
        self.d7_bn1     = nn.BatchNorm2d(64)
        self.dense7_c2  = nn.Conv2d(in_channels=64+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d7_bn2     = nn.BatchNorm2d(96)
        self.dense7_c3  = nn.Conv2d(in_channels=96+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d7_bn3     = nn.BatchNorm2d(128)

        #up sampling
        self.UP3 = nn.Sequential(
             nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2)
        ) 

        ###dense8 32 -> 32
        self.dense8_c1  = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))#concat
        self.d8_bn1     = nn.BatchNorm2d(64)
        self.dense8_c2  = nn.Conv2d(in_channels=64+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d8_bn2     = nn.BatchNorm2d(96)
        self.dense8_c3  = nn.Conv2d(in_channels=96+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d8_bn3     = nn.BatchNorm2d(128)

        self.UP4 = nn.Sequential(
             nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=(2,2),stride=(2,2)),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.2)
        ) 

        ###dense9 32 -> 32
        self.dense9_c1  = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))#concat
        self.d9_bn1     = nn.BatchNorm2d(64)
        self.dense9_c2  = nn.Conv2d(in_channels=64+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d9_bn2     = nn.BatchNorm2d(96)
        self.dense9_c3  = nn.Conv2d(in_channels=96+32,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.d9_bn3     = nn.BatchNorm2d(128)


        self.last_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=(1, 1),
                stride=1,

            ),
            nn.Sigmoid()
        )
        
        


    def forward(self,x):
        #活性化関数(activation function)の定義
        conv_activation = torch.nn.LeakyReLU(0.2)
        #print("x: ",x.shape)
        first  = self.first_conv(x)
        #print("first: ",first.shape)
        dense1_1 = self.dense1_c1(conv_activation( self.d1_bn1(first)))
        dense1_2 = self.dense1_c2(conv_activation( self.d1_bn2(torch.cat((dense1_1,first),dim=1) ) ))
        dense1_3 = self.dense1_c3(conv_activation( self.d1_bn3(torch.cat((dense1_2,dense1_1,first),dim=1) ) ))
        D1       = self.DS1(dense1_3)#([2, 32, 256, 64])
     
        #print("D1: ",D1.shape)
        dense2_1 = self.dense2_c1(conv_activation( self.d2_bn1(D1)))
        dense2_2 = self.dense2_c2(conv_activation( self.d2_bn2(torch.cat((dense2_1,D1),dim=1) ) ))
        dense2_3 = self.dense2_c3(conv_activation( self.d2_bn3(torch.cat((dense2_2,dense2_1,D1),dim=1) ) ))
        D2       = self.DS2(dense2_3)#([2, 32, 128, 32])
     
        #print("D2: ",D2.shape)
        dense3_1 = self.dense3_c1(conv_activation( self.d3_bn1(D2)))
        dense3_2 = self.dense3_c2(conv_activation( self.d3_bn2(torch.cat((dense3_1,D2),dim=1) ) ))
        dense3_3 = self.dense3_c3(conv_activation( self.d3_bn3(torch.cat((dense3_2,dense3_1,D2),dim=1) ) ))
        D3 = self.DS3(dense3_3)
                
        #print("D3: ",D3.shape)
        dense4_1 = self.dense4_c1(conv_activation( self.d4_bn1(D3)))
        dense4_2 = self.dense4_c2(conv_activation( self.d4_bn2(torch.cat((dense4_1,D3),dim=1) ) ))
        dense4_3 = self.dense4_c3(conv_activation( self.d4_bn3(torch.cat((dense4_2,dense4_1,D3),dim=1) ) ))
        D4 = self.DS4(dense4_3)

        #print("D4: ",D4.shape)
        dense5_1 = self.dense5_c1(conv_activation( self.d5_bn1(D4)))
        dense5_2 = self.dense5_c2(conv_activation( self.d5_bn2(torch.cat((dense5_1,D4),dim=1) ) ))
        dense5_3 = self.dense5_c3(conv_activation( self.d5_bn3(torch.cat((dense5_2,dense5_1,D4),dim=1) ) ))
        
        UP1      = self.UP1(dense5_3)
        UP1      = torch.cat((UP1,dense4_3),dim=1)#([2, 64, 256, 64])
        #print("UP1: ",UP1.shape)
        dense6_1 = self.dense6_c1(conv_activation( self.d6_bn1(UP1)))
        dense6_2 = self.dense6_c2(conv_activation( self.d6_bn2(torch.cat((dense6_1,UP1),dim=1) ) ))
        dense6_3 = self.dense6_c3(conv_activation( self.d6_bn3(torch.cat((dense6_2,dense6_1,UP1),dim=1) ) ))

        UP2      = self.UP2(dense6_3)
        UP2      = torch.cat((UP2,dense3_3),dim=1) 
        #print("UP2: ",UP2.shape)
        dense7_1 = self.dense7_c1(conv_activation( self.d7_bn1(UP2)))
        dense7_2 = self.dense7_c2(conv_activation( self.d7_bn2(torch.cat((dense7_1,UP2),dim=1) ) ))
        dense7_3 = self.dense7_c3(conv_activation( self.d7_bn3(torch.cat((dense7_2,dense7_1,UP2),dim=1) ) ))

        UP3      = self.UP3(dense7_3)
        UP3      = torch.cat((UP3,dense2_3),dim=1)
        #print("UP3: ",UP3.shape)
        dense8_1 = self.dense8_c1(conv_activation( self.d8_bn1(UP3)))
        dense8_2 = self.dense8_c2(conv_activation( self.d8_bn2(torch.cat((dense8_1,UP3),dim=1) ) ))
        dense8_3 = self.dense8_c3(conv_activation( self.d8_bn3(torch.cat((dense8_2,dense8_1,UP3),dim=1) ) ))
        
        UP4      = self.UP4(dense8_3)
        UP4      = torch.cat((UP4,dense1_3),dim=1)

        #print("UP3: ",UP4.shape)
        dense9_1 = self.dense9_c1(conv_activation( self.d9_bn1(UP4)))
        dense9_2 = self.dense9_c2(conv_activation( self.d9_bn2(torch.cat((dense9_1,UP4),dim=1) ) ))
        dense9_3 = self.dense9_c3(conv_activation( self.d9_bn3(torch.cat((dense9_2,dense9_1,UP4),dim=1) ) ))
        
        Mask   = self.last_conv(dense9_3)
        #print(Mask.shape)
        output  = torch.multiply(Mask,x)
        return output