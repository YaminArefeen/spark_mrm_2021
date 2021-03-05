#Code used to generate Figure 2 in the ISMRM abstract "Extending Scan-specific Artifact Reduction in K-space (SPARK) to Advanced Encoding and Reconstruction Schemes" (Yamin Arefeen et al.).  Note, results may not be exact, as the initialization of neural network weights is not standardized across different runs of the experiment.
#The script takes the following steps
#   -Load the grappa reconstructed kspace, fully sampled kspace, and object with the parameters for the reconstruction (acceleration factors, acs size, etc)
#   -Reformat the appropriate kspace (acs kspace and grappa kspace) to be inputted into the SPARK model
#   -Train a set of real and imaginary models for each coil
#   -Apply the kspace correction through the SPARK model
#   -Save the reesults for future comparison

import time

print('Importing libraries, defining helper functions, and loading the dataset... ',end='')
start = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy as sp
from utils import cfl
from utils import signalprocessing as sig

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#DEFINING HELPER FUNCTIONS AND SPARK MODEL
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fft3(x):
    return sig.fft(sig.fft(sig.fft(x,-3),-2),-1)

def ifft3(x):
    return sig.ifft(sig.ifft(sig.ifft(x,-3),-2),-1)

class SPARK_3D_net(nn.Module):
    def __init__(self,coils,kernelsize,acsx,acsy,acsz):
        super().__init__()
        self.acsx = acsx
        self.acsy = acsy
        self.acsz = acsz
        
        self.conv1 = nn.Conv3d(coils*2,coils*2,kernelsize,padding=1,bias = False)
        self.conv2 = nn.Conv3d(coils*2,coils,1,padding=0,bias=False)
        self.conv3 = nn.Conv3d(coils, coils*2, kernelsize, padding=1, bias=False)
        self.conv4 = nn.Conv3d(coils*2,coils*2,kernelsize,padding=1,bias = False)
        self.conv5 = nn.Conv3d(coils*2,coils,1,padding=0,bias=False)
        self.conv6 = nn.Conv3d(coils, coils*2, kernelsize, padding=1, bias=False)
        self.conv7 = nn.Conv3d(coils*2, coils, kernelsize, padding=1, bias=False)
        self.conv8 = nn.Conv3d(coils, coils//4, 1, padding=0, bias=False)
        self.conv9 = nn.Conv3d(coils//4, 1, kernelsize, padding=1, bias=False)    

    def nl(self,inp):
        return inp + F.relu((inp-1)/2) + F.relu((-inp-1)/2)    
        
    def forward(self, x):

        y = self.nl(self.conv1(x))
        y = self.nl(self.conv2(y))
        y = self.nl(self.conv3(y))
        y = x + y
        z = self.nl(self.conv4(y))
        z = self.nl(self.conv5(z))
        z = self.nl(self.conv6(z))
        out = z  + y
        out = self.conv9(self.nl(self.conv8(self.nl(self.conv7(out)))))

        loss_out = out[:,:,self.acsx[0]:self.acsx[-1]+1,self.acsy[0]:self.acsy[-1]+1,self.acsz[0]:self.acsz[-1]+1]

        return out, loss_out
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Setting Parameters and loading the dataset
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Loading fully sampled kspace, grappa reconstructed kspace, and parmaters
kspace       = np.transpose(cfl.readcfl('data/kspaceFull'),(3,0,1,2))
kspaceGrappa = np.transpose(cfl.readcfl('data/kspaceGrappa'),(3,0,1,2))
for3dspark   = sp.io.loadmat('data/forSpark.mat')['for3Dspark'][0][0]

[C,M,N,P] = kspace.shape

Rx = for3dspark[0][0][0]
Ry = for3dspark[0][0][1]
Rz = for3dspark[0][0][2]

acsx = for3dspark[1][0][0]
acsy = for3dspark[1][0][2]
acsz = for3dspark[1][0][2]

#Defining some SPARK parameters
normalizationflag = 1       #If we want to normalize datasets befoe training SPARK model
measuredReplace   = 1       #If we want to replace measured data (as well as ACS)
iterations        = 1000    #Number of iterations to train each spark network
learningRate      = .002    #Learning rate to train each spark network
kernelsize        = 3       #3 x 3 x 3 convolutional kernels in the SPARK network

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print('Elapsed Time is %.3f seconds' % (time.time()-start))

print('Acquisition Parameters Parameters: ')
print('  Dimensions:   %d x %d x %d x %d' %(C,M,N,P))
print('  Acceleration: %d x %d x %d' % (Rx,Ry,Rz))
print('  ACS Sizes:    %d x %d x %d' % (acsx,acsy,acsz))

print('SPARK Training Parameters: ')
print('  Iterations: %d' % iterations)
print('  Stepsize:   %.3f' % learningRate)
print('  Kernel:     %d' % kernelsize)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Generating Zerofilled ACS, grappa recon, and reformatting kspace for SPARK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Generating zerofilled ACS, grappa replaced recon, and reformatting kspace for SPARK... ',end='')
start = time.time()

#-Generating zerofilled acs kspace from the fully sampled kspace.  This will be used as the reference from which we will train the SPARK
acsregionX = np.arange(M//2 - acsx // 2,M//2 + acsx//2) 
acsregionY = np.arange(N//2 - acsy // 2,N//2 + acsy//2) 
acsregionZ = np.arange(P//2 - acsz // 2,P//2 + acsz//2) 

kspaceAcsZerofilled = np.zeros(kspace.shape,dtype = complex)
kspaceAcsZerofilled[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1] = \
    kspace[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1]

#-Generating ACS replaced GRAPPA recon for comparisons later on
tmp = np.copy(kspaceGrappa)
tmp[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1] = \
    kspace[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1]

#-Reformatting kspace for SPARK
kspaceAcsCrop   = kspace[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1] 
kspaceAcsGrappa = kspaceGrappa[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1] 
kspaceAcsDifference = kspaceAcsCrop - kspaceAcsGrappa

acs_difference_real = np.real(kspaceAcsDifference)
acs_difference_imag = np.imag(kspaceAcsDifference)

kspace_grappa = np.copy(kspaceGrappa)
kspace_grappa_real  = np.real(kspace_grappa)
kspace_grappa_imag  = np.imag(kspace_grappa)
kspace_grappa_split = np.concatenate((kspace_grappa_real, kspace_grappa_imag), axis=0)

#Normalizing data if specified 
chan_scale_factors_real = np.zeros(C,dtype = 'float')
chan_scale_factors_imag = np.zeros(C,dtype = 'float')

if(normalizationflag):
    scale_factor_input  = 1/np.amax(np.abs(kspace_grappa_split))
    kspace_grappa_split *= scale_factor_input

for c in range(C):
    if(normalizationflag):
        scale_factor_real = 1/np.amax(np.abs(acs_difference_real[c,:,:,:]))
        scale_factor_imag = 1/np.amax(np.abs(acs_difference_imag[c,:,:,:]))
    else:
        scale_factor_real = 1
        scale_factor_imag = 1

    chan_scale_factors_real[c] = scale_factor_real
    chan_scale_factors_imag[c] = scale_factor_imag

    acs_difference_real[c,:,:,:] *= scale_factor_real
    acs_difference_imag[c,:,:,:] *= scale_factor_imag

acs_difference_real = np.expand_dims(np.expand_dims(acs_difference_real,axis=1), axis=1)
acs_difference_imag = np.expand_dims(np.expand_dims(acs_difference_imag,axis=1), axis=1)

kspace_grappa_split = torch.unsqueeze(torch.from_numpy(kspace_grappa_split),axis = 0)
kspace_grappa_split = kspace_grappa_split.to(device, dtype=torch.float)

acs_difference_real = torch.from_numpy(acs_difference_real)
acs_difference_real = acs_difference_real.to(device, dtype=torch.float)

acs_difference_imag = torch.from_numpy(acs_difference_imag)
acs_difference_imag = acs_difference_imag.to(device, dtype=torch.float)

print('Elapsed Time is %.3f seconds' % (time.time()-start))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Training the SPARK Networks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
#-Training the real spark networks
real_models      = {}
real_model_names = []

criterion   = nn.MSELoss()

realLoss = np.zeros((iterations,C)) 

for c in range(0,C):
    model_name = 'model'+ 'C' + str(c) + 'r'
    model = SPARK_3D_net(coils=C,kernelsize=kernelsize,acsx=acsregionX,acsy=acsregionY,acsz=acsregionZ)

    model.to(device)

    print('Training {}'.format(model_name))
    start = time.time()
    
    optimizer = optim.Adam(model.parameters(),lr=learningRate)
    running_loss = 0

    for epoch in range(iterations):
        optimizer.zero_grad()

        _,loss_out = model(kspace_grappa_split)
        loss = criterion(loss_out,acs_difference_real[c,:,:,:,:,:]) 
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        realLoss[epoch,c] = running_loss;
        
        if(epoch == 0):
            print('   Starting Loss: %.10f' % running_loss)

    real_model_names.append(model_name)
    real_models.update({model_name:model})
    
    print('   Ending Loss:   %.10f' % (running_loss))
    print('   Training Time: %.3f seconds' % (time.time() - start))
    
#-Training the imaginary spark networks
imag_models      = {}
imag_model_names = []

criterion   = nn.MSELoss()

imagLoss = np.zeros((iterations,C)) 

for c in range(0,C):
    model_name = 'model'+ 'C' + str(c) + 'i'
    model = SPARK_3D_net(coils=C,kernelsize=kernelsize,acsx=acsregionX,acsy=acsregionY,acsz=acsregionZ)

    model.to(device)

    print('Training {}'.format(model_name))
    start = time.time()
    
    optimizer = optim.Adam(model.parameters(),lr=learningRate)
    running_loss = 0

    for epoch in range(iterations):
        optimizer.zero_grad()

        _,loss_out = model(kspace_grappa_split)
        loss = criterion(loss_out,acs_difference_imag[c,:,:,:,:,:]) 
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        imagLoss[epoch,c] = running_loss;
        
        if(epoch == 0):
            print('   Starting Loss: %.10f' % running_loss)
            
    imag_model_names.append(model_name)
    imag_models.update({model_name:model})
    
    print('   Ending Loss:   %.10f' % (running_loss))
    print('   Training Time: %.3f seconds' % (time.time() - start))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Applying the SPARK correction
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Performing coil-by-coil correction... ', end = '')
start = time.time()

kspaceCorrected = np.zeros((C,M,N,P),dtype = complex)

for c in range(0,C):
    #Perform reconstruction coil by coil
    model_namer = 'model' + 'C' + str(c) + 'r'
    model_namei = 'model' + 'C' + str(c) + 'i'

    real_model = real_models[model_namer]
    imag_model = imag_models[model_namei]

    correctionr = real_model(kspace_grappa_split)[0].cpu().detach().numpy()
    correctioni = imag_model(kspace_grappa_split)[0].cpu().detach().numpy() 
    
    kspaceCorrected[c,:,:,:] = correctionr[0,0,:,:,:]/chan_scale_factors_real[c] + \
        1j * correctioni[0,0,:,:,:] / chan_scale_factors_imag[c] + kspaceGrappa[c,:,:,:]
    
print('Elapsed Time is %.3f seconds' % (time.time()-start))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Perofrming ACS replacement and ifft/rsos coil combine reconstruction
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Performing ACS replacement, ifft, and rsos coil combination... ', end = '')
start = time.time()

#ACS replaced
kspaceCorrectedReplaced    = np.copy(kspaceCorrected)

kspaceCorrectedReplaced[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1] = \
    kspace[:,acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,acsregionZ[0]:acsregionZ[acsz-1]+1]

#Sampled Data replacement
if(measuredReplace):
    kspaceCorrectedReplaced[:,::Rx,::Ry,::Rz] = kspace[:,::Rx,::Ry,::Rz]
    
#Perform IFFT and coil combine
truth  = sig.rsos(ifft3(kspace),-4)
grappa = sig.rsos(ifft3(tmp),-4)
spark  = sig.rsos(ifft3(kspaceCorrectedReplaced),-4)
print('Elapsed Time is %.3f seconds' % (time.time()-start))

#~~~~~~~~~~~~~~~~~~
#Saving the Results
#~~~~~~~~~~~~~~~~~~
print('Saving results... ', end = '')
start = time.time()
results = {'groundTruth': np.squeeze(truth),
           'grappa': np.squeeze(grappa),
           'spark': np.squeeze(spark),
           'Ry': Ry,
           'Rz': Rz,
           'acsy': acsy,
           'acsz': acsz,           
           'Iterations': iterations,
           'learningRate': learningRate,
           'realLoss':realLoss,
           'imagLoss':imagLoss}

sp.io.savemat('results.mat', results, oned_as='row')
print('Elapsed Time is %.3f seconds' % (time.time()-start))
