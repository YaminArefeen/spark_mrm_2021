import torch
import torch.nn as nn
import torch.nn.functional as F

class SPARK_Net(nn.Module):
    def __init__(self,coils,kernelsize,acs):
        super().__init__()
        self.acs = acs

        self.conv1 = nn.Conv2d(coils*2, coils*2, kernelsize, padding=1, bias=False)
        self.conv2 = nn.Conv2d(coils*2, coils, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(coils, coils*2, kernelsize, padding=1, bias=False)
        self.conv4 = nn.Conv2d(coils*2, coils, kernelsize, padding=1, bias=False)
        self.conv5 = nn.Conv2d(coils, coils//4, 1, padding=0, bias=False)
        self.conv6 = nn.Conv2d(coils//4, 1, kernelsize, padding=1, bias=False)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        z = x + y
        z = F.relu(self.conv4(z))
        z = F.relu(self.conv5(z))
        out = self.conv6(z)

        loss_out = out[:,:,:,self.acs]

        return out, loss_out

class SPARK_Netv2(nn.Module):
    def __init__(self,coils,kernelsize,acsx,acsy,naliniflag = 0):
        super().__init__()
        self.acsx = acsx
        self.acsy = acsy
        self.naliniflag = naliniflag
        
        self.conv1 = nn.Conv2d(coils*2, coils*2, kernelsize, padding=1, bias=False)
        self.conv2 = nn.Conv2d(coils*2, coils, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(coils, coils*2, kernelsize, padding=1, bias=False)
        self.conv4 = nn.Conv2d(coils*2, coils, kernelsize, padding=1, bias=False)
        self.conv5 = nn.Conv2d(coils, coils//4, 1, padding=0, bias=False)
        self.conv6 = nn.Conv2d(coils//4, 1, kernelsize, padding=1, bias=False)
        
    def naliniRelu(self,inp):
        #An attempt at implementing Nalini's custom nonlinearity, from "Joint Frequency- and Image-Space Learning for Fourier Imaging"
        return inp + F.relu((inp-1)/2) + F.relu((-inp-1)/2)       

    def forward(self, x):
        
        if(self.naliniflag):
            y = self.naliniRelu(self.conv1(x))
            y = self.naliniRelu(self.conv2(y))
            y = self.naliniRelu(self.conv3(y))
            z = x + y
            z = self.naliniRelu(self.conv4(z))
            z = self.naliniRelu(self.conv5(z))
            out = self.conv6(z)
        else:
            y = F.relu(self.conv1(x))
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            z = x + y
            z = F.relu(self.conv4(z))
            z = F.relu(self.conv5(z))
            out = self.conv6(z)
        
        loss_out = out[:,:,self.acsx[0]:self.acsx[-1]+1,self.acsy[0]:self.acsy[-1]+1]

        return out, loss_out


class SPARK_Netv3(nn.Module):
    def __init__(self,coils,kernelsize,trainingMask):
        super().__init__()
        self.trainingMask = trainingMask

        self.conv1 = nn.Conv2d(coils*2, coils*2, kernelsize, padding=1, bias=False)
        self.conv2 = nn.Conv2d(coils*2, coils, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(coils, coils*2, kernelsize, padding=1, bias=False)
        self.conv4 = nn.Conv2d(coils*2, coils, kernelsize, padding=1, bias=False)
        self.conv5 = nn.Conv2d(coils, coils//4, 1, padding=0, bias=False)
        self.conv6 = nn.Conv2d(coils//4, 1, kernelsize, padding=1, bias=False)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        z = x + y
        z = F.relu(self.conv4(z))
        z = F.relu(self.conv5(z))
        out = self.conv6(z)

        loss_out = out * self.trainingMask

        return out, loss_out

