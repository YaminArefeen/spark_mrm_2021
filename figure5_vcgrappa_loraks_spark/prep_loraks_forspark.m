% In this script, prepare loraks for SPARK reconstruction.  In particular:
%   a)  kspace-LORAKS grappa (without acs replacement for SPARK)
%   b)  kspace-fully sampled (to use as ACS comparison as well as ground truth)
%   c)  kspace-LORAKS grappa with acs replacement, for comparison
%   e)  Acquisition Parameters: acs size, acceleration factors
addpath matlabutils/
load data/img_grappa_32chan

%% Loading the fully-sampled coil image and generate some preliminaries
IMG = rot90(IMG,-1);
[N(1), N(2), num_chan, num_cycle] = size(IMG);
kspace_full  = fft2c( IMG );

Rz = 7;     % accl in kz
Ry = 1;     % accl in ky 

use_vc = 1;

num_acs = [30,N(2)-2];          % acs region size 

acs_x1 = floor(N(1)/2) - floor(num_acs(1)/2) + 1; % inclusive
acs_y1 = floor(N(2)/2) - floor(num_acs(2)/2) + 1; % inclusive
acs_x2 = floor(N(1)/2) + floor(num_acs(1)/2);     % inclusive
acs_y2 = floor(N(2)/2) + floor(num_acs(2)/2);     % inclusive

truth = rsos(IMG,3);

%% Performing undersampling
kspace_acs      = zeross([N, num_chan]);
kspace_sampled  = zeross([N, num_chan]);

kspace_acs(acs_x1:acs_x2, acs_y1:acs_y2, :) = kspace_full(acs_x1:acs_x2, acs_y1:acs_y2, :);
kspace_sampled(1:Rz:end,1:Ry:end,:)         = kspace_full(1:Rz:end,1:Ry:end,:);

%% LORAKS with ACS replacement
r_S = 200;       % rank constraint
R   = 2;           % local k-space radius

pcg_tol  = 1e-6;  % pcg tolerance to terminate
pcg_iter = 100;  % pcg max iterations


Mask_acs = zeross([N, num_chan]);
Mask_acs(1+end/2-num_acs(1)/2:end/2+num_acs(1)/2 + 1, 1+end/2-num_acs(2)/2:end/2+num_acs(2)/2 + 1, :) = 1;

Mask_sampled = zeross([N, num_chan]);
Mask_sampled(1:Rz:end, 1:Ry:end, :) = 1;

mask = reshape(Mask_acs + Mask_sampled, N(1), N(2), num_chan);
mask = mask>0;

% undersampled data
Kspace_sampled = fft2c( IMG ) .* mask;

kData = reshape(Kspace_sampled,N(1),N(2),[]);
kMask = reshape(mask,N(1),N(2),[]);

if(use_vc)
    % virtual coils
    kDatavirt = flipdim(flipdim(conj(kData),1),2);
    kMaskvirt = flipdim(flipdim(kMask,1),2);
    % combined actual and virtual coils
    kData = reshape([kData(:);kDatavirt(:)],N(1),N(2),[]);
    kMask = reshape([kMask(:);kMaskvirt(:)],N(1),N(2),[]);
end

N1 = size(kData,1);
N2 = size(kData,2);
Nc = size(kData,3);

% Find loraks annhilating filters 
Nic = calib_loraks( kData, kMask, R, r_S );

S.type = '()';
S.subs{:} = find(~vect(ifftshift(kMask)));

phi = @(x) subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(subsasgn(zeros(N1,N2,Nc),S,x)))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);
b = -subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(reshape(ifftshift(kData),[N1,N2,Nc])))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);

% PCG recon
tic
    [z, flag, res, iter, resvec] = pcg(@(x)  phi(x), b, pcg_tol, pcg_iter);
toc

A = @(x) fftshift(ifftshift(kData) + subsasgn(zeros(N1,N2,Nc),S,x));
k = A(z);

if(use_vc)
    loraks_replaced_coils = reshape(ifft2c(reshape(k(1:N1*N2*Nc/2),[N1,N2,Nc/2])), [N, num_chan]);
else
    loraks_replaced_coils = reshape(ifft2c(reshape(k,[N1,N2,Nc])), [N, num_chan]);
end

%% LORAKS without acs replacement
% r_S = 100;         % rank constraint: between [0,1], 1-> full rank
% R   = 2;              % local k-space radius

pcg_tol = 1e-6;     % pcg tolerance to terminate
pcg_iter = 100;      % pcg max iterations

% undersampled data
mask = zeros([N,num_chan]);
mask(1:Rz:end,1:Ry:end,:) = 1;
mask(1+end/2-num_acs(1)/2:end/2+num_acs(1)/2 + 1, 1+end/2-num_acs(2)/2:end/2+num_acs(2)/2 + 1, :) = 1;

Kspace_sampled = double( fft2c(IMG) .* mask );

kData = reshape(Kspace_sampled,N(1),N(2),[]);
kMask = reshape(mask,N(1),N(2),[]);

if(use_vc)
    % virtual coils
    kDatavirt = flipdim(flipdim(conj(kData),1),2);
    kMaskvirt = flipdim(flipdim(kMask,1),2);
    % combined actual and virtual coils
    kData = reshape([kData(:);kDatavirt(:)],N(1),N(2),[]);
    kMask = reshape([kMask(:);kMaskvirt(:)],N(1),N(2),[]);
end

N1 = size(kData,1);
N2 = size(kData,2);
Nc = size(kData,3);

% Find loraks annhilating filters
tic
    Nic = calib_loraks( kData, kMask, R, r_S );
toc

% % undersampled data
mask = zeros([N,num_chan]);
mask(1:Rz:end,1:Ry:end,:) = 1;

Kspace_sampled = double( fft2c(IMG) .* mask );

kData = reshape(Kspace_sampled,N(1),N(2),[]);
kMask = reshape(mask,N(1),N(2),[]);

if(use_vc)
    kDatavirt = flipdim(flipdim(conj(kData),1),2);
    kMaskvirt = flipdim(flipdim(kMask,1),2);

    kData = reshape([kData(:);kDatavirt(:)],N(1),N(2),[]);
    kMask = reshape([kMask(:);kMaskvirt(:)],N(1),N(2),[]);
end

S.type = '()';
S.subs{:} = find(~vect(ifftshift(kMask)));

phi = @(x) subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(subsasgn(zeros(N1,N2,Nc),S,x)))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);
b = -subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(reshape(ifftshift(kData),[N1,N2,Nc])))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);

% PCG recon
tic
    [z, flag, res, iter, resvec] = pcg(@(x)  phi(x), b, pcg_tol, pcg_iter);
toc

disp(['flag: ', num2str(flag), ' iter: ' num2str(iter), ' res: ', num2str(res)])

A = @(x) fftshift(ifftshift(kData) + subsasgn(zeros(N1,N2,Nc),S,x));
k = A(z);
 
if(use_vc)
    loraks_coils = reshape(ifft2c(reshape(k(1:N1*N2*Nc/2),[N1,N2,Nc/2])), [N, num_chan]);
else
    loraks_coils = reshape(ifft2c(reshape(k(1:N1*N2*Nc),[N1,N2,Nc])), [N, num_chan]);
end

%% Show and save some results
n = @(x) abs(x) / max(abs(x(:)));

loraks_replaced = rsos(loraks_replaced_coils,3);
loraks          = rsos(loraks_coils,3);

figure; imshow([n(truth) n(loraks_replaced) n(loraks)]);
fprintf("ACS REPLACED RMSE:    %.2f\n",rmse(truth,loraks_replaced));
fprintf("NO ACS REPLACED RMSE: %.2f\n",rmse(truth,loraks));

%-Saving for SPARK
forspark_loraks.kspace_full            = kspace_full;
forspark_loraks.kspace_loraks          = fft2c(loraks_coils);
forspark_loraks.kspace_svc_replaced    = fft2c(loraks_replaced_coils);

forspark_loraks.Rx     = Rz;
forspark_loraks.Ry     = Ry;
forspark_loraks.acsx   = num_acs(1);
forspark_loraks.acsy   = num_acs(2);

save('forspark/rforspark_loraks.mat','forspark_loraks')