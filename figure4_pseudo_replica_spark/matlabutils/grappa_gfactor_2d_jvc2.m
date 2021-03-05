function [ img_grappa, mask, mask_acs, image_weights, g_fnl ] = grappa_gfactor_2d_jvc2( kspace_sampled, kspace_acs, Rx, Ry, acs_size, kernel_size, lambda_tik, subs, delx, dely, weights_p)
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



if nargin < 9
    % k-space sampling offset of each coil
    delx = zeros([1,num_chan]);
    dely = zeros([1,num_chan]);
end

num_acsX = acs_size(1);                % acs size
num_acsY = acs_size(2);                % acs size

% sampling and acs masks
mask_acs = zeros([N, num_chan]);
mask_acs(1+end/2-num_acsX/2:end/2+num_acsX/2 + 1, 1+end/2-num_acsY/2:end/2+num_acsY/2 + 1, :) = 1;

mask = zeros([N, num_chan]);
for c = 1:num_chan
    mask(delx(c)+1:Rx:end, dely(c)+1:Ry:end, c) = 1;
end

kernel_hsize = (kernel_size-1)/2;

% pad_size = kernel_hsize .* [Rx,Ry];
pad_size = kernel_size .* [Rx,Ry];
N_pad = N + 2*pad_size;


% k-space limits for training:
ky_begin = 1 + Ry * kernel_hsize(2);       % first kernel center point that fits acs region 
ky_end = num_acsY - Ry * kernel_hsize(2) + 1;   % last kernel center point that fits acs region 
ky_end = ky_end - max(dely);

kx_begin = 1 + Rx * kernel_hsize(1);            % first kernel center point that fits acs region 
kx_end = num_acsX - Rx * kernel_hsize(1) + 1;           % last kernel center point that fits acs region 
kx_end = kx_end - max(delx); 


% k-space limits for recon:
Ky_begin = 1 + Ry * kernel_hsize(2);       % first kernel center point that fits acs region 
Ky_end = N_pad(2) - Ry * kernel_hsize(2);      % last kernel center point that fits acs region 
Ky_end = Ky_end - max(dely);

Kx_begin = 1 + Rx * kernel_hsize(1);            % first kernel center point that fits acs region 
Kx_end = N_pad(1) - Rx * kernel_hsize(1);           % last kernel center point that fits acs region 
Kx_end = Kx_end - max(delx);



% count the no of kernels that fit in acs 
ind = 1;

for ky = ky_begin : ky_end
    for kx = kx_begin : kx_end
        ind = ind + 1;        
    end
end

num_ind = ind;
     

kspace_acs_crop = kspace_acs(1+end/2-num_acsX/2:end/2+num_acsX/2 + 1, 1+end/2-num_acsY/2:end/2+num_acsY/2 + 1, :);

Rhs = zeross([num_ind, num_chan, Rx*Ry-1]);
acs = zeross([kernel_size, num_chan]);
Acs = zeross([num_ind, prod(kernel_size) * num_chan]);

disp(['ACS mtx size: ', num2str(size(Acs))])


ind = 1;

for ky = ky_begin : ky_end
    for kx = kx_begin : kx_end

        for c = 1:num_chan
            acs(:,:,c) = kspace_acs_crop(delx(c)+kx-kernel_hsize(1)*Rx : Rx : delx(c)+kx+kernel_hsize(1)*Rx, dely(c)+ky-kernel_hsize(2)*Ry : Ry : dely(c)+ky+kernel_hsize(2)*Ry, c);
        end
        
        Acs(ind,:) = acs(:);
        
        idx = 1;
        for ry = 1:Ry-1
            
            for c = 1:num_chan
                Rhs(ind,c,idx) = kspace_acs_crop(delx(c)+kx, dely(c)+ky-ry, c);
            end
            
            idx = idx + 1;
        end  
        
        for rx = 1:Rx-1
            for ry = 0:Ry-1
                
                for c = 1:num_chan
                    Rhs(ind,c,idx) = kspace_acs_crop(delx(c)+kx-rx, dely(c)+ky-ry, c);
                end
                
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

weights = zeros([prod(kernel_size) * num_chan, num_chan, Rx*Ry-1]);

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
data = zeross([kernel_size, num_chan]);


for ky = Ky_begin : Ry : Ky_end
    for kx = Kx_begin : Rx : Kx_end

        for c = 1:num_chan
            data(:,:,c) = kspace_recon(delx(c)+kx-kernel_hsize(1)*Rx : Rx : delx(c)+kx+kernel_hsize(1)*Rx, dely(c)+ky-kernel_hsize(2)*Ry : Ry : dely(c)+ky+kernel_hsize(2)*Ry, c);         
        end
        
        
        idx = 1;
        for ry = 1:Ry-1
            Wdata = Weights(:,:,idx) * data(:);

            for c = 1:num_chan
                kspace_recon(delx(c)+kx, dely(c)+ky-ry, c) = Wdata(c);
            end
            
            idx = idx + 1;
        end    

        for rx = 1:Rx-1
            for ry = 0:Ry-1
                Wdata = Weights(:,:,idx) * data(:);

                for c = 1:num_chan
                    kspace_recon(delx(c)+kx-rx, dely(c)+ky-ry, c) = Wdata(c);
                end
                
                idx = idx + 1;
            end    
        end
        
    end
end

kspace_recon = kspace_recon(1+pad_size(1):end-pad_size(1), 1+pad_size(2):end-pad_size(2), :);


if subs
    % subsititute sampled & acs data
    kspace_recon = kspace_recon .* (~mask & ~mask_acs) + kspace_sampled .* (~mask_acs) + kspace_acs .* mask_acs;
end

img_grappa = ifft2c(kspace_recon);
 





% image space grappa weights
if nargout > 3

    image_weights = zeross([N, num_chan, num_chan]);


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



    g_fnl = 0;

    if (nargin == 11) && ( sum(weights_p(:)) ~=0 )

        % coil combined g-factor 

        img_weights = image_weights / (Ry * Rx);

        num_actual = size(weights_p, 3);

        g_cmb = zeross([N, num_chan]);

        for c = 1:num_chan

            weights_coil = squeeze( img_weights(:,:,c,1:num_actual) );

            g_cmb(:,:,c) = sum(weights_p .* weights_coil, 3);

        end

        g_comb = sqrt( sum(g_cmb .* conj( g_cmb ), 3) );

        p_comb = sqrt( sum(weights_p .* conj( weights_p ), 3) );

        g_fnl = g_comb ./ p_comb;

    end

end

end

