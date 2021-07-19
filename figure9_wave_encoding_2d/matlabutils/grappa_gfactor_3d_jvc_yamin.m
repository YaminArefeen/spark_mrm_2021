function [ img_grappa, mask, mask_acs, image_weights, g_fnl ] = grappa_gfactor_3d_jvc_yamin(kspace_sampled, kspace_acs, Ry, Rz, ... 
    acs_size, kernel_size, lambda_tik, subs, dely, delz, weights_p,verbose)
%GRAPPA_1D Summary of this function goes here
%   Detailed explanation goes here

if nargin < 6
    kernel_size = [3,3,3];
end

if nargin < 7
    lambda_tik = eps;
end

if nargin < 8
    subs = 1;
end

if nargin < 12
    verbose = 1;
end




[N(1), N(2), N(3), num_chan] = size(kspace_sampled);

num_acsX = acs_size(1);                % acs size
num_acsY = acs_size(2);                % acs size
num_acsZ = acs_size(3);                % acs size


if nargin < 10
    % k-space sampling offset of each coil
    delz = zeros([1,num_chan]);
end

if nargin < 9
    dely = zeros([1,num_chan]);
end


 

% sampling and acs masks
mask_acs = zeros([N, num_chan]);
mask_acs(1+end/2-num_acsX/2:end/2+num_acsX/2 + 1, 1+end/2-num_acsY/2:end/2+num_acsY/2 + 1, 1+end/2-num_acsZ/2:end/2+num_acsZ/2 + 1, :) = 1;

mask = zeros([N, num_chan]);
for c = 1:num_chan
    mask(:, dely(c)+1:Ry:end, delz(c)+1:Rz:end, c) = 1;
end


kernel_hsize = (kernel_size-1)/2;

pad_size = kernel_hsize .* [1,Ry,Rz];
N_pad = N + 2*pad_size;


% k-space limits for training:
kz_begin = 1 + Rz * kernel_hsize(3);                    % first kernel center point that fits acs region 
kz_end = num_acsZ - Rz * kernel_hsize(3) + 1;           % last kernel center point that fits acs region 
kz_end = kz_end - max(delz); 

ky_begin = 1 + Ry * kernel_hsize(2);                    % first kernel center point that fits acs region 
ky_end = num_acsY - Ry * kernel_hsize(2) + 1;           % last kernel center point that fits acs region 
ky_end = ky_end - max(dely);

kx_begin = 1 + kernel_hsize(1);                    % first kernel center point that fits acs region 
kx_end = num_acsX - kernel_hsize(1) + 1;           % last kernel center point that fits acs region 



% k-space limits for recon:
Kz_begin = 1 + Rz * kernel_hsize(3);                    % first kernel center point that fits acs region 
Kz_end = N_pad(3) - Rz * kernel_hsize(3);               % last kernel center point that fits acs region 
Kz_end = Kz_end - max(delz);

Ky_begin = 1 + Ry * kernel_hsize(2);                    % first kernel center point that fits acs region 
Ky_end = N_pad(2) - Ry * kernel_hsize(2);               % last kernel center point that fits acs region 
Ky_end = Ky_end - max(dely);

Kx_begin = 1 + kernel_hsize(1);                    % first kernel center point that fits acs region 
Kx_end = N_pad(1) - kernel_hsize(1);               % last kernel center point that fits acs region 
 


% count the no of kernels that fit in acs 
ind = 1;

for kz = kz_begin : kz_end
    for ky = ky_begin : ky_end
        for kx = kx_begin : kx_end
            ind = ind + 1;        
        end
    end
end

num_ind = ind;
     

kspace_acs_crop = kspace_acs(1+end/2-num_acsX/2:end/2+num_acsX/2 + 1, 1+end/2-num_acsY/2:end/2+num_acsY/2 + 1, 1+end/2-num_acsZ/2:end/2+num_acsZ/2 + 1, :);


Rhs = zeros([num_ind, num_chan, Rz*Ry-1]);
acs = zeros([kernel_size, num_chan]);
Acs = zeros([num_ind, prod(kernel_size) * num_chan]);

if(verbose)
disp(['ACS mtx size: ', num2str(size(Acs))])
end

ind = 1;


for kz = kz_begin : kz_end
    for ky = ky_begin : ky_end
        for kx = kx_begin : kx_end

            for c = 1:num_chan
                acs(:,:,:,c) = kspace_acs_crop( kx-kernel_hsize(1) : kx+kernel_hsize(1), dely(c)+ky-kernel_hsize(2)*Ry : Ry : dely(c)+ky+kernel_hsize(2)*Ry, delz(c)+kz-kernel_hsize(3)*Rz : Rz : delz(c)+kz+kernel_hsize(3)*Rz, c);
            end

            Acs(ind,:) = acs(:);

            idx = 1;
            for ry = 1:Ry-1

                for c = 1:num_chan
                    Rhs(ind,c,idx) = kspace_acs_crop(kx, dely(c)+ky-ry, delz(c)+kz, c);
                end

                idx = idx + 1;
            end  

            for rz = 1:Rz-1
                for ry = 0:Ry-1

                    for c = 1:num_chan
                        Rhs(ind,c,idx) = kspace_acs_crop(kx, dely(c)+ky-ry, delz(c)+kz-rz, c);
                    end

                    idx = idx + 1;
                end    
            end

            ind = ind + 1;
            
        end
    end
end  


if lambda_tik
    [u,s,v] = svd(Acs, 'econ');

    s_inv = diag(s); 
    
    if(verbose)
    disp(['condition number: ', num2str(max(abs(s_inv)) / min(abs(s_inv)))])
    end
    
    s_inv = conj(s_inv) ./ (abs(s_inv).^2 + lambda_tik);

    Acs_inv = v * diag(s_inv) * u';
end


% estimate kernel weights

weights = zeros([prod(kernel_size) * num_chan, num_chan, Rz*Ry-1]);

for r = 1:Rz*Ry-1
    if(verbose)
    disp(['Kernel group : ', num2str(r)])
    end
    
    for c = 1:num_chan

        if ~lambda_tik
            weights(:,c,r) = Acs \ Rhs(:,c,r);
        else
            weights(:,c,r) = Acs_inv * Rhs(:,c,r);
        end

    end
end



% recon undersampled data

Weights = permute(weights, [2,1,3]);

kspace_recon = padarray(kspace_sampled, [pad_size, 0]);
data = zeros([kernel_size, num_chan]);


for kz = Kz_begin : Rz : Kz_end
    
    if(verbose)
    disp([num2str((kz - Kz_begin + Rz) / Rz), ' / ', num2str((Kz_end - Kz_begin + Rz) / Rz)])
    end
    
    for ky = Ky_begin : Ry : Ky_end
        for kx = Kx_begin : Kx_end

            for c = 1:num_chan
                data(:,:,:,c) = kspace_recon(kx-kernel_hsize(1) : kx+kernel_hsize(1), dely(c)+ky-kernel_hsize(2)*Ry : Ry : dely(c)+ky+kernel_hsize(2)*Ry, delz(c)+kz-kernel_hsize(3)*Rz : Rz : delz(c)+kz+kernel_hsize(3)*Rz, c);         
            end


            idx = 1;
            for ry = 1:Ry-1
                Wdata = Weights(:,:,idx) * data(:);

                for c = 1:num_chan
                    kspace_recon(kx, dely(c)+ky-ry, delz(c)+kz, c) = Wdata(c);
                end

                idx = idx + 1;
            end    

            for rz = 1:Rz-1
                for ry = 0:Ry-1
                    Wdata = Weights(:,:,idx) * data(:);

                    for c = 1:num_chan
                        kspace_recon(kx, dely(c)+ky-ry, delz(c)+kz-rz, c) = Wdata(c);
                    end

                    idx = idx + 1;
                end    
            end
        
        end
    end
end

kspace_recon = kspace_recon(1+pad_size(1):end-pad_size(1), 1+pad_size(2):end-pad_size(2), 1+pad_size(3):end-pad_size(3), :);


if subs
    % subsititute sampled & acs data
    kspace_recon = kspace_recon .* (~mask & ~mask_acs) + kspace_sampled .* (~mask_acs) + padarray(kspace_acs, ([N, num_chan] - size(kspace_acs))/2);
end

img_grappa = mifft3(kspace_recon);
 



if nargout > 3
    % NOT IMPLEMENTED YET

    % image space grappa weights

    image_weights = zeros([N, num_chan, num_chan]);


    for c = 1:num_chan

        idx = 1; 

        image_weights(delx(c) + 1+end/2, dely(c) + 1+end/2, c, c) = 1;

        for ry = 1:Ry-1

            w = weights(:, c, idx);

            W = reshape(w, [kernel_size, num_chan]);

            for coil = 1:num_chan
                image_weights( delx(coil) + 1+end/2 - kernel_hsize(1)*Rx :Rx: delx(coil) + 1+end/2 + kernel_hsize(1)*Rx, ...
                    dely(coil) + ry + 1+end/2 - kernel_hsize(2)*Ry :Ry: dely(coil) + ry + 1+end/2 + kernel_hsize(2)*Ry, coil, c) = W(:,:,coil);
            end


            idx = idx + 1;

        end    

        for rx = 1:Rx-1
            for ry = 0:Ry-1

                w = weights(:, c, idx);

                W = reshape(w, [kernel_size, num_chan]);

                for coil = 1:num_chan
                    image_weights( delx(coil) + rx + 1+end/2 - kernel_hsize(1)*Rx :Rx: delx(coil) + rx + 1+end/2 + kernel_hsize(1)*Rx, ...
                        dely(coil) + ry + 1+end/2 - kernel_hsize(2)*Ry :Ry: dely(coil) + ry + 1+end/2 + kernel_hsize(2)*Ry, coil, c) = W(:,:,coil);
                end

                idx = idx + 1;

            end    
        end

    end


    image_weights = flipdim( flipdim( ifft2c2( image_weights ), 1 ), 2 ) * sqrt(prod(N));
end


 

if (nargin == 11) && (nargout > 4)

    % coil combined g-factor 

    img_weights = image_weights / (Ry * Rx);

    num_actual = size(weights_p, 3);
    
    g_cmb = zeros([N, num_chan]);

    for c = 1:num_chan

        weights_coil = squeeze( img_weights(:,:,c,1:num_actual) );

        g_cmb(:,:,c) = sum(weights_p .* weights_coil, 3);

    end

    g_comb = sqrt( sum(g_cmb .* conj( g_cmb ), 3) );

    p_comb = sqrt( sum(weights_p .* conj( weights_p ), 3) );

    g_fnl = g_comb ./ p_comb;
  
end



end

