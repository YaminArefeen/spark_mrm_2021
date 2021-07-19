import numpy as xp
import cupy as cp
import scipy as sp

import scipy.io
import matplotlib.pyplot as plt

def fft(x,ax):
    '''
    Performs a centered, orthonormal, forward fourier transform along the specified axes
    Ixputs:
        x   - Object which we will be fourier transforming 
        ax  - Axis along which we want to take the centered fourier transform 
    Outputs:
        out - Fourier transformed object
    '''
    xp = cp.get_array_module(x)
    return xp.fft.fftshift(xp.fft.fft(xp.fft.ifftshift(x,axes = ax),axis = ax,norm='ortho'),axes=ax)

def ifft(x,ax):
    '''
    Performs a centered, orthonormal, forward fourier transform along the specified axes
    Ixputs:
        x   - Object which we will be fourier transforming 
        ax  - Axis along which we want to take the centered fourier transform 
    Outputs:
        out - Fourier transformed object
    '''
    xp = cp.get_array_module(x)
    return xp.fft.fftshift(xp.fft.ifft(xp.fft.ifftshift(x,axes = ax),axis = ax,norm='ortho'),axes=ax)

def fft2c(x):
    '''
    Performs a 2D centered, orthonormal, forward fourier transform along the last two axis of the image.
    Inputs:
        x    - Object to be fourier transformed, the last two dimensions should be the image dimensions Outputs:
        out  - Forward Fourier Transformed object
    '''
    xp = cp.get_array_module(x)
    return fft(fft(x,-1),-2)

def ifft2c(x):
    '''
    Performs a 2D centered, orthonormal, inverse fourier transform along the last two axis of the image.
    Inputs:
        x    - Object to be fourier transformed, the last two dimensions should be the image dimensions Outputs:
        out  - Inverse Fourier Transformed object
    '''
    xp = cp.get_array_module(x)
    return ifft(ifft(x,-1),-2)

def rmse(original,comparison):
    '''
    Computes the normalized root mean squared error between an original (ground truth) object and a comparison.  As long as the object are of the same dimension, this function will vectorize and compute the desired value
    '''
    return xp.sqrt(xp.sum(xp.square(xp.abs(original-comparison))))/xp.sqrt(xp.sum(xp.square(xp.abs(original))))

def nor(x):
    '''
    Normalizes a input numpy vector where its maximum value is set to 1.
    '''
    return xp.abs(x) / xp.max(x.flatten())

def mosaic(img, num_row = 1, num_col = 1, fig_num=0, clim=[0,1], fig_title='', num_rot=0, fig_size=(18, 16)):    
    fig = plt.figure(fig_num, figsize=fig_size)
    fig.patch.set_facecolor('black')

    img = xp.abs(img);
    img = img.astype(float)
        
    if img.ndim < 3:
        img = xp.rot90(img, k=num_rot, axes=(0,1))
        img_res = img
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)
        title_str = fig_title
        plt.savefig(title_str + '.png')

    else: 
        img = xp.rot90(img, k=num_rot,axes=(1,2))
        
        if img.shape[0] != (num_row * num_col):
            print('sizes do not match')    
        else:   
            img_res = xp.zeros((img.shape[1]*num_row, img.shape[2]*num_col))
            
            idx = 0
            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r*img.shape[1] : (r+1)*img.shape[1], c*img.shape[2] : (c+1)*img.shape[2]] = img[idx,:,:]
                    idx = idx + 1
               
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)
        plt.title(fig_title, color='white')
        title_str = fig_title
        plt.savefig(title_str + '.png')

def grappa(samples, acs, Rx, Ry, num_acs, shift_x, shift_y, kernel_size=xp.array([3,3]), lambda_tik=0,verbose = 0):
    # Set Initial Parameters
    #------------------------------------------------------------------------------------------
    [num_chan, N1, N2] = samples.shape
    N = xp.array([N1, N2]).astype(int)
    
    acs_start_index_x = N1//2 - num_acs[0]//2 #inclusive
    acs_start_index_y = N2//2 - num_acs[1]//2 #inclusive
    acs_end_index_x = xp.int(xp.ceil(N1/2)) + num_acs[0]//2
    acs_end_index_y = xp.int(xp.ceil(N2/2)) + num_acs[1]//2
    
    kspace_sampled = xp.zeros(samples.shape, dtype=samples.dtype)
    kspace_sampled[:] = samples[:]
    
    kspace_acs = xp.zeros(acs.shape, dtype=acs.dtype)
    kspace_acs[:] = acs[:]
    
    kspace_acs_crop = xp.zeros([num_chan, num_acs[0], num_acs[1]], dtype=acs.dtype)
    kspace_acs_crop[:,:,:] = kspace_acs[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y]
  
    #Kernel Side Size
    kernel_hsize = (kernel_size // 2).astype(int)

    #Padding
    pad_size = (kernel_hsize * [Rx,Ry]).astype(int)
    N_pad = N + 2 * pad_size

    # Beginning/End indices for kernels in the acs region
    ky_begin_index = (Ry*kernel_hsize[1]).astype(int)
    ky_end_index = (num_acs[1] - Ry*kernel_hsize[1] - 1 - xp.amax(shift_y)).astype(int)

    kx_begin_index = (Rx*kernel_hsize[0]).astype(int)
    kx_end_index = (num_acs[0] - Rx*kernel_hsize[0] - 1 - xp.amax(shift_x)).astype(int)

    # Beginning/End indices for kernels in the full kspace
    Ky_begin_index = (Ry*kernel_hsize[1]).astype(int)
    Ky_end_index = (N_pad[1] - Ry*kernel_hsize[1] - 1).astype(int)

    Kx_begin_index = (Rx*kernel_hsize[0]).astype(int)
    Kx_end_index = (N_pad[0] - Rx*kernel_hsize[0] - 1).astype(int)

    # Count the number of kernels that fit the acs region
    ind = 0
    for i in range(ky_begin_index, ky_end_index+1):
        for j in range(kx_begin_index, kx_end_index+1):
            ind +=1

    num_kernels = ind

    # Initialize right hand size and acs_kernel matrices
    target_data = xp.zeros([num_kernels, num_chan, Rx, Ry], dtype=samples.dtype)
    kernel_data = xp.zeros([num_chan, kernel_size[0], kernel_size[1]], dtype=samples.dtype)
    acs_data = xp.zeros([num_kernels, kernel_size[0] * kernel_size[1] * num_chan], dtype=samples.dtype)

    # Get kernel and target data from the acs region
    #------------------------------------------------------------------------------------------
    if(verbose):
        print('Collecting kernel and target data from the acs region')
        print('------------------------------------------------------------------------------------------')
    kernel_num = 0
    
    for ky in range(ky_begin_index, ky_end_index + 1):
        if(verbose):
            print('ky: ' + str(ky))
        for kx in range(kx_begin_index, kx_end_index + 1):
            # Get kernel data
            for nchan in range(0,num_chan):
                kernel_data[nchan, :, :] = kspace_acs_crop[nchan, 
                                                           shift_x[nchan] + kx - (kernel_hsize[0]*Rx): shift_x[nchan] + kx + (kernel_hsize[0]*Rx)+1: Rx, 
                                                           shift_y[nchan] + ky - (kernel_hsize[1]*Ry): shift_y[nchan] + ky + (kernel_hsize[1]*Ry)+1: Ry]

            acs_data[kernel_num, :] = kernel_data.flatten()

            # Get target data
            for rx in range(0,Rx):
                for ry in range(0,Ry):
                    if rx != 0 or ry != 0:
                        for nchan in range(0,num_chan):
                            target_data[kernel_num,:,rx,ry] = kspace_acs_crop[:,
                                                                              shift_x[nchan] + kx - rx,
                                                                              shift_y[nchan] + ky - ry]

            # Move to the next kernel
            kernel_num += 1
    if(verbose):        
        print()

    # Tikhonov regularization
    #------------------------------------------------------------------------------------------
    U, S, Vh = sp.linalg.svd(acs_data, full_matrices=False)
   
    if(verbose):
        print('Condition number: ' + str(xp.max(xp.abs(S))/xp.min(xp.abs(S))))
        print()
    
    S_inv = xp.conjugate(S) / (xp.square(xp.abs(S)) + lambda_tik)
    acs_data_inv = xp.transpose(xp.conjugate(Vh)) @ xp.diag(S_inv) @ xp.transpose(xp.conjugate(U));

    # Get kernel weights
    #------------------------------------------------------------------------------------------
    if(verbose):
        print('Getting kernel weights')
        print('------------------------------------------------------------------------------------------')
    kernel_weights = xp.zeros([num_chan, kernel_size[0] * kernel_size[1] * num_chan, Rx, Ry], dtype=samples.dtype)

    for rx in range(0,Rx):
        if(verbose):
            print('rx: ' + str(rx))
        for ry in range(0,Ry):
            if(verbose):
                print('ry: ' + str(ry))
            if rx != 0 or ry != 0:
                for nchan in range(0,num_chan):
                    if(verbose):
                        print('Channel: ' + str(nchan+1))
                    if lambda_tik == 0:
                        kernel_weights[nchan, :, rx, ry], resid, rank, s = xp.linalg.lstsq(acs_data,target_data[:, nchan, rx, ry], rcond=None)
                    else:
                        kernel_weights[nchan, :, rx, ry] = acs_data_inv @ target_data[:, nchan, rx, ry]
                        
    if(verbose):
        print()

    # Reconstruct unsampled points
    #------------------------------------------------------------------------------------------
    if(verbose):
        print('Reconstructing unsampled points')
        print('------------------------------------------------------------------------------------------')
    kspace_recon = xp.pad(kspace_sampled, ((0, 0), (pad_size[0],pad_size[0]), (pad_size[1],pad_size[1])), 'constant')
    data = xp.zeros([num_chan, kernel_size[0] * kernel_size[1]], dtype=samples.dtype)

    for ky in range(Ky_begin_index, Ky_end_index+1, Ry):
        if(verbose):
            print('ky: ' + str(ky))
        for kx in range(Kx_begin_index, Kx_end_index+1, Rx):

            for nchan in range(0,num_chan):
                data[nchan, :] = (kspace_recon[nchan,
                                               shift_x[nchan] + kx - (kernel_hsize[0]*Rx): shift_x[nchan] + kx + (kernel_hsize[0]*Rx)+1: Rx, 
                                               shift_y[nchan] + ky - (kernel_hsize[1]*Ry): shift_y[nchan] + ky + (kernel_hsize[1]*Ry)+1: Ry]).flatten()


            for rx in range(0,Rx):
                for ry in range(0,Ry):
                    if rx != 0 or ry != 0:
                        for nchan in range(0,num_chan):
                            interpolation = xp.dot(kernel_weights[nchan, :, rx, ry] , data.flatten())
                            kspace_recon[nchan, shift_x[nchan] + kx - rx, shift_y[nchan] + ky - ry] = interpolation

    # Get the image back
    #------------------------------------------------------------------------------------------
    kspace_recon = kspace_recon[:, pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]  
    img_grappa = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))
    
    if(verbose):
        print()
        print('GRAPPA reconstruction complete.')
    
    return kspace_recon, img_grappa

def f2(x):
    #flattens just the first two dimensions of a numpy matrix, hardcodd to assume the numpy matrix has 4 dimensions
    return xp.reshape(x,(x.shape[0]*x.shape[1],x.shape[-2],x.shape[-1]))

def mip(x,axis):
    '''
    Compute a maximum intensity projection image across axis axis
    '''
    return xp.amax(xp.abs(x),axis = axis)

def rsos(x,axis):
    '''
    Computes root sum of square along the specified axis
    '''
    return xp.sqrt(xp.sum(xp.square(xp.abs(x)),axis = axis))
