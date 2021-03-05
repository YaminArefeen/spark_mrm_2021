function [ img_grappa, mask, mask_acs ] = grappa_2d( kspace_sampled, kspace_acs, Rx, Ry, acs_size, kernel_size, lambda_tik, subs )
%GRAPPA_1D Summary of this function goes here
%   Detailed explanation goes here

if nargin < 6
    kernel_size = [3,3];
end

if nargin < 7
    lambda_tik = eps;
end

if nargin < 8
    subs = 1;
end


[N(1), N(2), num_chan] = size(kspace_sampled);

num_acsX = acs_size(1);                % acs size
num_acsY = acs_size(2);                % acs size
 

% sampling and acs masks
mask = zeros(N);
mask_acs = zeros(N);

mask(1:Rx:end,1:Ry:end) = 1;
mask_acs(1+end/2-num_acsX/2:end/2+num_acsX/2 + 1, 1+end/2-num_acsY/2:end/2+num_acsY/2 + 1) = 1;


kernel_hsize = (kernel_size-1)/2;

pad_size = kernel_hsize .* [Rx,Ry];
N_pad = N + 2*pad_size;


% k-space limits for training:
ky_begin = 1 + Ry * kernel_hsize(2);       % first kernel center point that fits acs region 
ky_end = num_acsY - Ry * kernel_hsize(2) + 1;   % last kernel center point that fits acs region 


kx_begin = 1 + Rx * kernel_hsize(1);            % first kernel center point that fits acs region 
kx_end = num_acsX - Rx * kernel_hsize(1) + 1;           % last kernel center point that fits acs region 
 

% k-space limits for recon:
Ky_begin = 1 + Ry * kernel_hsize(2);       % first kernel center point that fits acs region 
Ky_end = N_pad(2) - Ry * kernel_hsize(2);      % last kernel center point that fits acs region 


Kx_begin = 1 + Rx * kernel_hsize(1);            % first kernel center point that fits acs region 
Kx_end = N_pad(1) - Rx * kernel_hsize(1);           % last kernel center point that fits acs region 



% count the no of kernels that fit in acs 
ind = 1;

for ky = ky_begin : ky_end
    for kx = kx_begin : kx_end
        ind = ind + 1;        
    end
end

num_ind = ind;
     

kspace_acs_crop = kspace_acs(1+end/2-num_acsX/2:end/2+num_acsX/2 + 1, 1+end/2-num_acsY/2:end/2+num_acsY/2 + 1, :);


Rhs = zeros([num_ind, num_chan, Rx*Ry-1]);
Acs = zeros([num_ind, prod(kernel_size) * num_chan]);

disp(['ACS mtx size: ', num2str(size(Acs))])


ind = 1;

for ky = ky_begin : ky_end
    for kx = kx_begin : kx_end

        acs = kspace_acs_crop(kx-kernel_hsize(1)*Rx:Rx:kx+kernel_hsize(1)*Rx, ky-kernel_hsize(2)*Ry:Ry:ky+kernel_hsize(2)*Ry, :);

        Acs(ind,:) = acs(:);

        idx = 1;
        for ry = 1:Ry-1
            Rhs(ind,:,idx) = kspace_acs_crop(kx, ky-ry, :);
            idx = idx + 1;
        end    

        for rx = 1:Rx-1
            for ry = 0:Ry-1
                Rhs(ind,:,idx) = kspace_acs_crop(kx-rx, ky-ry, :);
                idx = idx + 1;
            end    
        end
        
        ind = ind + 1;
    end
end


if lambda_tik
    [u,s,v] = svd(Acs, 'econ');

    s_inv = diag(s); 
    
    disp(['condition number: ', num2str(max(abs(s_inv)) / min(abs(s_inv)))])
    
    s_inv = conj(s_inv) ./ (abs(s_inv).^2 + lambda_tik);

    Acs_inv = v * diag(s_inv) * u';
end


% estimate kernel weights

weights = zeros([prod(kernel_size) * num_chan, num_chan, Ry-1]);

for r = 1:Rx*Ry-1
    disp(['Kernel group : ', num2str(r)])

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


for ky = Ky_begin : Ry : Ky_end
    for kx = Kx_begin : Rx : Kx_end

        data = kspace_recon(kx-kernel_hsize(1)*Rx:Rx:kx+kernel_hsize(1)*Rx, ky-kernel_hsize(2)*Ry:Ry:ky+kernel_hsize(2)*Ry, :);                

        idx = 1;
        for ry = 1:Ry-1
            kspace_recon(kx, ky-ry, :) = Weights(:, :, idx) * data(:);
            idx = idx + 1;
        end    

        for rx = 1:Rx-1
            for ry = 0:Ry-1
                kspace_recon(kx-rx, ky-ry, :) = Weights(:, :, idx) * data(:);
                idx = idx + 1;
            end    
        end
        
    end
end

kspace_recon = kspace_recon(1+pad_size(1):end-pad_size(1), 1+pad_size(2):end-pad_size(2), :);

if subs
    % subsititute sampled & acs data
    kspace_recon = kspace_recon .* repmat((~mask & ~mask_acs), [1,1,num_chan]) + kspace_sampled .* repmat(~mask_acs, [1,1,num_chan]) + kspace_acs;
end

img_grappa = ifft2(kspace_recon);
 
  

end

