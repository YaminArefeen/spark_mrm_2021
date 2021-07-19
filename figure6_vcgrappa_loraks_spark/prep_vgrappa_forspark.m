% In this script, prepare vc-grappa for SPARK reconstruction.  In particular, preparethe following:
%   a)  kspace-vc grappa (without acs replacement for SPARK)
%   b)  kspace-fully sampled (to use as ACS comparison as well as ground truth)
%   c)  kspace-vc grappa with acs replacement, for comparison
%   e)  Acquisition Parameters: acs size, acceleration factors
addpath matlabutils/
load data/img_grappa_32chan

%% Loading the fully-sampled coil image and generate some preliminaries
IMG = rot90(IMG,-1);
[N(1), N(2), num_chan, num_cycle] = size(IMG);
kspace_full  = fft2c( IMG );

Rz = 7;     % accl in kz
Ry = 1;     % accl in ky 

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

%% Perform SPARSE GRAPPA
[k2,k1] = meshgrid(0:N(2)-1, 0:N(1)-1);

fdx = fftshift(-1 + exp(2*pi*1i*k1/N(1)));
fdy = fftshift(-1 + exp(2*pi*1i*k2/N(2)));

Fdx = repmat(fdx, [1,1,num_chan]);
Fdy = repmat(fdy, [1,1,num_chan]);

fig_num = 3;

lambda_tik = 1e-7;          % regularization parameter for calibration
kernel_size = [3,3];        % use odd kernel size

Xhatp = kspace_sampled + (kspace_sampled==0) .* kspace_acs;

W = Xhatp~=0;

[Img_grappa_DvX, ~, ~] = grappa_gfactor_2d_jvc2( kspace_sampled .* Fdx, kspace_acs .* Fdx, Rz, Ry, num_acs, kernel_size, lambda_tik, 1 );
            
[Img_grappa_DhX, mask, mask_acs] = grappa_gfactor_2d_jvc2( kspace_sampled .* Fdy, kspace_acs .* Fdy, Rz, Ry, num_acs, kernel_size, lambda_tik, 1 );

Img_grappa = ifft2call( (conj(Fdx) .* fft2call(Img_grappa_DvX) + conj(Fdy) .* fft2call(Img_grappa_DhX)) ./ (abs(Fdx).^2 + abs(Fdy).^2 + eps) .* (1-W) + Xhatp );

%% Perform VC-sparse GRAPPA
lambda_tik = 1e-7;      % tikhonov reg for kernel estimation in initial joint grappa
lambda_vc = 1e-5;       % tikhonov reg for kernel estimation in later jvc grappa iterations

fig_num = 10;

num_iter = 2;

num_acs = [30,N(2)-2];          % acs size for kernel estimation
num_acs_vc = N-2;               % use entire k-space of current recon for vc kernel estimation 

kernel_size = [3,7];            % use odd kernel size
kernel_size_vc = [3,13];         % use odd kernel size


% virtual coils for the sampled data:
Kspace_sampled_vc = cat( 3, kspace_sampled, fft2c(conj(ifft2c(kspace_sampled))) );

% set small entries due to fft in missing virtual coil k-space to zero:
Kspace_sampled_vc( abs(Kspace_sampled_vc) < 1e-9 ) = 0;

% vc for acs:
Kspace_acs_vc = cat(3, kspace_acs, fft2c(conj(ifft2c(kspace_acs))));

msk_sampled = squeeze(Kspace_sampled_vc(:,:,1:num_chan)~=0);
msk_sampled_vc = squeeze(Kspace_sampled_vc(:,:,1+num_chan:end)~=0);


Delz_vc = zeros(1, num_chan);
Dely_vc = zeros(1, num_chan);

% find the delta shift relative to first sampling pattern in the virtual coils
for vc = 1:num_chan*num_cycle    
    sum_best = sum(sum(abs(msk_sampled(:,:,1) - msk_sampled_vc(:,:,vc))));
    
    for rz = 0:Rz-1
        for ry = 0:Ry-1
            sum_now = sum(sum(abs(msk_sampled(:,:,1) - circshift( msk_sampled_vc(:,:,vc), [-rz, -ry] ))));
            
            if sum_now < sum_best
                Delz_vc(vc) = rz;
                Dely_vc(vc) = ry;
                sum_best = sum_now;
            end
        end
    end
end


DelZ = cat(2, Delz_vc*0, Delz_vc);
DelY = cat(2, Dely_vc*0, Dely_vc);

% k-space offset of each phase-cycle:
[DelZ(1:num_chan:end).', DelY(1:num_chan:end).']


tic
for n = 1:num_iter
    disp(['Iter: ', num2str(n)])
    
    if n == 1

        % initial vc grappa recon
        [Img_vcgrappa, mask, mask_acs] = grappa_gfactor_2d_jvc2( Kspace_sampled_vc, Kspace_acs_vc, Rz, Ry, num_acs, kernel_size, lambda_tik, 1, DelZ, DelY );

        rmse_vcgrappa = rmse( rsos(Img_vcgrappa(:,:,1:num_chan),3), truth )

        Img_grappa = Img_vcgrappa(:,:,1:num_chan);

        mosaic( rsos(Img_vcgrappa(:,:,1:num_chan),3), 1, 1, fig_num, ['svc1 Grappa: ', num2str(rmse_vcgrappa)], [0,1], 0),  
    else
        % joint virtual coil grappa recon
        Kspace_grappa_vc = fft2c( cat(3, Img_grappa, conj(Img_grappa) ) );
        
        Kspace_acs_vc = zeross([N, num_chan*2]);

        Kspace_acs_vc(1+end/2-num_acs_vc(1)/2:end/2+num_acs_vc(1)/2 + 1, 1+end/2-num_acs_vc(2)/2:end/2+num_acs_vc(2)/2 + 1, :) = ...
            Kspace_grappa_vc(1+end/2-num_acs_vc(1)/2:end/2+num_acs_vc(1)/2 + 1, 1+end/2-num_acs_vc(2)/2:end/2+num_acs_vc(2)/2 + 1, :);

        Im_vcgrappa = grappa_gfactor_2d_jvc2( Kspace_sampled_vc, Kspace_acs_vc, Rz, Ry, num_acs_vc, kernel_size_vc, lambda_vc, 0, DelZ, DelY );
        
        % substitute sampled & acs data
        Kspace_vcgrappa = fft2c(Im_vcgrappa(:,:,1:num_chan));


        Img_grappa = ifft2c( Kspace_vcgrappa .* (~mask(:,:,1:num_chan) & ~mask_acs(:,:,1:num_chan)) ... 
            + kspace_sampled .* (~mask_acs(:,:,1:num_chan)) + kspace_acs );
        
        Img_Grappa_vc = reshape(Img_grappa, [N, num_chan]);

    
        rmse_vcgrappa = rmse( rsos(Img_Grappa_vc,3), truth )
    
        mosaic( rsos(Img_Grappa_vc, 3), 1, 1, fig_num+1, ['svc2 Grappa: ', num2str(rmse_vcgrappa)], [0,1], 0 ),  
    end
end
toc


%% Visualize prepare for SPARK
n = @(x) abs(x) / max(abs(x(:)));

svcgrappa_acsreplaced = rsos(Img_Grappa_vc,3);
svcgrappa             = rsos(ifft2call(Kspace_vcgrappa),3);

figure; imshow([n(truth) n(svcgrappa_acsreplaced) n(svcgrappa)]);
fprintf("ACS REPLACED RMSE:    %.2f\n",rmse(truth,svcgrappa_acsreplaced));
fprintf("NO ACS REPLACED RMSE: %.2f\n",rmse(truth,svcgrappa));

%-Saving for SPARK
forspark_svc.kspace_full            = kspace_full;
forspark_svc.kspace_svc             = Kspace_vcgrappa;
forspark_svc.kspace_svc_replaced    = fft2call(Img_Grappa_vc);

forspark_svc.Rx     = Rz;
forspark_svc.Ry     = Ry;
forspark_svc.acsx   = num_acs(1);
forspark_svc.acsy   = num_acs(2);

save('forspark/forspark_svc.mat','forspark_svc')