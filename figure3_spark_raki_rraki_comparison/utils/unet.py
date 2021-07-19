import torch
import torch.nn as nn

class convBlock(nn.Module):
    def __init__(self,inChannels,outChannels,instancenormflag):
        '''
        Defines a convolutional block for a section of the UNET

        in channels    3 x 3 conv + relu      out channels  3 x 3 conv + relu   out channels
        |                                     |                                 |
        |               ---- >                |             ---->               |
        |                                     |                                 |
        |                                     |                                 |
        '''
        super().__init__()
        self.inChannels  = inChannels
        self.outChannels = outChannels
        self.instancenormflag = instancenormflag

        self.inorm = nn.InstanceNorm2d(outChannels)

        #First 3 x 3 convolution from in channels to out channels
        self.conv1 = nn.Conv2d(inChannels,outChannels,kernel_size = 3,padding = 1)
        #Second 3 x 3 convolution from out channels to out channels
        self.conv2 = nn.Conv2d(outChannels,outChannels,kernel_size = 3,padding = 1)
        #ReLu nonlienarity to be applied after both convolutions
        self.relu  = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        if(self.instancenormflag):
            out = self.inorm(out)
        out = self.relu(out)
        out = self.conv2(out)
        if(self.instancenormflag):
            out = self.inorm(out)

        out = self.relu(out)
        return out

class unet(nn.Module):
    def __init__(self,inChannels = 1,outChannels = 1,channels = 32,numlayers = 2,inormflag = 0,acs = 0):
        '''
        inChannels (int) - Number of input channels to the unet model
        outchannels(int) - Number of output channel to the unet model
        channels   (int) - Number of output channels for the first convolution block
        numlayers  (int) - Depth in the unet model
        '''
        super().__init__()

        self.acs = acs

        #Down sampling through max pool
        self.maxpool    = nn.MaxPool2d(kernel_size = 2)

        #Downsampling layer of the unet
        self.downsample = nn.ModuleList([convBlock(inChannels,channels,inormflag)])
        curChannels = channels #Records the current number of channels at each layer when making unet
        for i in range(numlayers-1):
            self.downsample.append(convBlock(curChannels,curChannels*2,inormflag))
            curChannels = curChannels * 2

        #Center convolution layer of the unet
        self.centerconv = convBlock(curChannels,curChannels,inormflag)

        #Upsampling layer of the unet
        self.upsample = nn.ModuleList([])
        for i in range(numlayers-1):
            self.upsample.append(convBlock(curChannels*2,curChannels//2,inormflag))
            curChannels = curChannels // 2

        self.upsample.append(convBlock(curChannels*2,curChannels,inormflag))

        #Define the two one by one convolutions at the end
        self.endconv1 = nn.Conv2d(curChannels,curChannels//2,kernel_size = 1)
        curChannels   = curChannels//2

        self.endconv2 = nn.Conv2d(curChannels,1,kernel_size = 1)

        
    def forward(self,x):
        stack = [] #Stack to keep track of downsampled outputs so we can keep and append to upsample

        #Apply the downsampling layer
        out = x
        for i in range(len(self.downsample)):
            out = self.downsample[i](out)
            stack.append(out)
            out = nn.functional.max_pool2d(out,kernel_size = 2)
            #out = self.maxpool(out)

        #Apply the center layer before applying upsampling
        out = self.centerconv(out)
        #Apply upsampling layer with concatonation from same place in downsampling layer
        for i in range(len(self.upsample)):
            out = nn.functional.interpolate(out,scale_factor = 2,mode = 'bilinear',align_corners=True)
            out = torch.cat((stack.pop(),out),1) #Element from the downsample layer that we wish to concatonate
            out = self.upsample[i](out)

        out = self.endconv1(out)
        out = self.endconv2(out)
       
        loss_out = out[:,:,:,self.acs]
        return out, loss_out

