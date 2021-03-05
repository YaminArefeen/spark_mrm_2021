%--------------------------------------------------------------------------
%% Load bssfp data: 4 phase cycles, 16 SVD compressed coils 
%--------------------------------------------------------------------------

% addpath /autofs/cluster/kawin/berkin/Matlab_Code_New/TOOLBOXES/Grappa_2D_Toolbox
% 
% 
% load /autofs/cluster/kawin/berkin/Matlab_Code_New/TOOLBOXES/Grappa_2D_Toolbox/img_raki
load img_grappa_32chan

% IMG = sq(img(:,:,11,:));
IMG = sq(img(:,:,:));

[N(1), N(2), num_chan, num_cycle] = size(IMG);


%--------------------------------------------------------------------------
%% coil compression
%--------------------------------------------------------------------------


num_acs = [42,N(2)-2];          % acs region size     

k_full = fft2c( IMG );

k_acs = k_full(1+end/2-num_acs(1)/2:end/2+num_acs(1)/2 + 1, 1+end/2-num_acs(2)/2:end/2+num_acs(2)/2 + 1, :);

mosaic(rsos(k_acs,3),1,1,10,'',[0,1e-3])


num_chan = 16;                  % num compressed coils

[k_acs_svd, cmp_mtx] = svd_compress2d(k_acs, num_chan, 1);

mosaic(rsos(k_acs_svd,3), 1, 1, 11, ['svd rmse: ', num2str(rmse(rsos(k_acs_svd,3), rsos(k_acs,3)))],[0,1e-3])


% apply compression
kspace_full = svd_apply2d(k_full, cmp_mtx);

IMG = ifft2c(kspace_full);

img_R1 = rsos(IMG, 3);



% %--------------------------------------------------------------------------
% %% grappa: 
% %--------------------------------------------------------------------------
% 
% Rz = 6;     % accl in kz
% Ry = 1;     % accl in ky 
% 
% 
% fig_num = 1;
% lambda_tik = 1e-6;              % tikhonov reg parameter for kernel calibration
% 
% 
% num_acs = [42,N(2)-2];          % acs region size     
% kernel_size = [3,13];            % use odd grappa kernel size
% 
% 
% delz = zeros(num_chan,1);       % staggering amount between cycles in kz
% dely = zeros(num_chan,1);       % staggering amount between cycles in ky
% 
% 
% kspace_acs = zeross([N, num_chan]);
% kspace_sampled = zeross([N, num_chan]);
% 
% kspace_acs(1+end/2-num_acs(1)/2:end/2+num_acs(1)/2 + 1, 1+end/2-num_acs(2)/2:end/2+num_acs(2)/2 + 1, :) = kspace_full(1+end/2-num_acs(1)/2:end/2+num_acs(1)/2 + 1, 1+end/2-num_acs(2)/2:end/2+num_acs(2)/2 + 1, :);
% kspace_sampled(1:Rz:end,1:Ry:end,:) = kspace_full(1:Rz:end,1:Ry:end,:);
%     
% tic
%     Img_Grappa = grappa_gfactor_2d_jvc2( kspace_sampled, kspace_acs, Rz, Ry, num_acs, kernel_size, lambda_tik, 1, delz, dely );    
% toc
% 
% 
% rmse_grappa = rmse( rsos(Img_Grappa,3), img_R1 )
% 
% mosaic( rsos(Img_Grappa, 3), 1, 1, fig_num, ['Grappa: ', num2str(rmse_grappa), '% rmse'], [0,1e-3], 180 ),  



%--------------------------------------------------------------------------
%% sparse grappa
%--------------------------------------------------------------------------
 
[k2,k1] = meshgrid(0:N(2)-1, 0:N(1)-1);

fdx = fftshift(-1 + exp(2*pi*1i*k1/N(1)));
fdy = fftshift(-1 + exp(2*pi*1i*k2/N(2)));

Fdx = repmat(fdx, [1,1,num_chan]);
Fdy = repmat(fdy, [1,1,num_chan]);

fig_num = 2;

lambda_tik = 1e-7;          % regularization parameter for calibration
kernel_size = [3,11];        % use odd kernel size


tic
    DvX = grappa_2d( kspace_sampled .* Fdx, kspace_acs .* Fdx, Rz, Ry, num_acs, kernel_size, lambda_tik );
    DhX = grappa_2d( kspace_sampled .* Fdy, kspace_acs .* Fdy, Rz, Ry, num_acs, kernel_size, lambda_tik );
toc


Xhatp = kspace_sampled + (kspace_sampled==0) .* kspace_acs;

W = Xhatp~=0;

X = ifft2call( (conj(Fdx) .* fft2call(DvX) + conj(Fdy) .* fft2call(DhX)) ./ (abs(Fdx).^2 + abs(Fdy).^2 + eps) .* (1-W) + Xhatp );

rmse_sgrappa = rmse( rsos(X,3), rsos(img_R1,3) )

mosaic( squeeze( rsos(X, 3) ), 1, 1, fig_num, ['s-Grappa: ', num2str(rmse_sgrappa), ' % rmse'], [0,1e-3], 180 )


 
% %--------------------------------------------------------------------------
% %% iterative virtual coil grappa: 
% %--------------------------------------------------------------------------
%  
% lambda_tik = 1e-6;      % tikhonov reg for kernel estimation in initial joint grappa
% lambda_vc = 1e-5;       % tikhonov reg for kernel estimation in later jvc grappa iterations
% 
% fig_num = 5;
% num_iter = 2;
% 
% num_acs = [42,N(2)-2];          % acs size for kernel estimation
% num_acs_vc = N-2;               % use entire k-space of current recon for vc kernel estimation 
% 
% kernel_size = [3,13];            % use odd kernel size
% kernel_size_vc = [9,9];         % use odd kernel size
% 
% 
% % virtual coils for the sampled data:
% Kspace_sampled_vc = cat( 3, kspace_sampled, fft2c(conj(ifft2c(kspace_sampled))) );
% 
% % set small entries due to fft in missing virtual coil k-space to zero:
% Kspace_sampled_vc( abs(Kspace_sampled_vc) < 1e-9 ) = 0;
% 
% % vc for acs:
% Kspace_acs_vc = cat(3, kspace_acs, fft2c(conj(ifft2c(kspace_acs))));
% 
% msk_sampled = squeeze(Kspace_sampled_vc(:,:,1:num_chan)~=0);
% msk_sampled_vc = squeeze(Kspace_sampled_vc(:,:,1+num_chan:end)~=0);
% 
% 
% Delz_vc = zeros(1, num_chan);
% Dely_vc = zeros(1, num_chan);
% 
% % find the delta shift relative to first sampling pattern in the virtual coils
% for vc = 1:num_chan*num_cycle    
%     sum_best = sum(sum(abs(msk_sampled(:,:,1) - msk_sampled_vc(:,:,vc))));
%     
%     for rz = 0:Rz-1
%         for ry = 0:Ry-1
%             sum_now = sum(sum(abs(msk_sampled(:,:,1) - circshift( msk_sampled_vc(:,:,vc), [-rz, -ry] ))));
%             
%             if sum_now < sum_best
%                 Delz_vc(vc) = rz;
%                 Dely_vc(vc) = ry;
%                 sum_best = sum_now;
%             end
%         end
%     end
% end
% 
% 
% DelZ = cat(2, Delz_vc*0, Delz_vc);
% DelY = cat(2, Dely_vc*0, Dely_vc);
% 
% % k-space offset of each phase-cycle:
% [DelZ(1:num_chan:end).', DelY(1:num_chan:end).']
% 
% 
% tic
% for n = 1:num_iter
%     disp(['Iter: ', num2str(n)])
%     
%     if n == 1
%         % initial vc grappa
%         [Img_vcgrappa, mask, mask_acs] = grappa_gfactor_2d_jvc2( Kspace_sampled_vc, Kspace_acs_vc, Rz, Ry, num_acs, kernel_size, lambda_tik, 1, DelZ, DelY );
% 
%         rmse_vcgrappa = rmse( rsos(Img_vcgrappa(:,:,1:num_chan),3), img_R1 )
% 
%         Img_grappa = Img_vcgrappa(:,:,1:num_chan);
%         
%         mosaic( rsos(Img_vcgrappa(:,:,1:num_chan),3), 1, 1, fig_num, ['vc Grappa: ', num2str(rmse_vcgrappa)], [0,1e-3], 180 ),  
%     else
%         % joint virtual coil grappa recon
%         Kspace_grappa_vc = fft2c( cat(3, Img_grappa, conj(Img_grappa) ) );
% 
%         Kspace_acs_vc = zeross([N, num_chan*2]);
% 
%         Kspace_acs_vc(1+end/2-num_acs_vc(1)/2:end/2+num_acs_vc(1)/2 + 1, 1+end/2-num_acs_vc(2)/2:end/2+num_acs_vc(2)/2 + 1, :) = ...
%             Kspace_grappa_vc(1+end/2-num_acs_vc(1)/2:end/2+num_acs_vc(1)/2 + 1, 1+end/2-num_acs_vc(2)/2:end/2+num_acs_vc(2)/2 + 1, :);
% 
%         Im_vcgrappa = grappa_gfactor_2d_jvc2( Kspace_sampled_vc, Kspace_acs_vc, Rz, Ry, num_acs_vc, kernel_size_vc, lambda_vc, 0, DelZ, DelY );
%         
%         % substitute sampled & acs data
%         Kspace_vcgrappa = fft2c(Im_vcgrappa(:,:,1:num_chan));
% 
%         Img_grappa = ifft2c( Kspace_vcgrappa .* (~mask(:,:,1:num_chan) & ~mask_acs(:,:,1:num_chan)) + kspace_sampled .* (~mask_acs(:,:,1:num_chan)) + kspace_acs );
%         
%         Img_Grappa_vc = reshape(Img_grappa, [N, num_chan]);
%     
%         rmse_vcgrappa = rmse( rsos(Img_Grappa_vc,3), img_R1 )
%     
%         mosaic( rsos(Img_Grappa_vc, 3), 1, 1, fig_num+2, ['vc Grappa: ', num2str(rmse_vcgrappa)], [0,1e-3], 180 ),  
%     end
% end
% toc
% 
% 
% 
% %--------------------------------------------------------------------------
% %% iterative virtual coil grappa: initialize with sparse-grappa
% %--------------------------------------------------------------------------
%  
% lambda_tik = 1e-7;      % tikhonov reg for kernel estimation in initial joint grappa
% lambda_vc = 1e-5;       % tikhonov reg for kernel estimation in later jvc grappa iterations
% 
% fig_num = 10;
% 
% use_sparse = 1;         % use sparse grappa to initialize iterations
% num_iter = 2;
% 
% num_acs = [42,N(2)-2];          % acs size for kernel estimation
% num_acs_vc = N-2;               % use entire k-space of current recon for vc kernel estimation 
% 
% kernel_size = [3,11];            % use odd kernel size
% kernel_size_vc = [9,9];         % use odd kernel size
% 
% 
% % virtual coils for the sampled data:
% Kspace_sampled_vc = cat( 3, kspace_sampled, fft2c(conj(ifft2c(kspace_sampled))) );
% 
% % set small entries due to fft in missing virtual coil k-space to zero:
% Kspace_sampled_vc( abs(Kspace_sampled_vc) < 1e-9 ) = 0;
% 
% % vc for acs:
% Kspace_acs_vc = cat(3, kspace_acs, fft2c(conj(ifft2c(kspace_acs))));
% 
% msk_sampled = squeeze(Kspace_sampled_vc(:,:,1:num_chan)~=0);
% msk_sampled_vc = squeeze(Kspace_sampled_vc(:,:,1+num_chan:end)~=0);
% 
% 
% Delz_vc = zeros(1, num_chan);
% Dely_vc = zeros(1, num_chan);
% 
% % find the delta shift relative to first sampling pattern in the virtual coils
% for vc = 1:num_chan*num_cycle    
%     sum_best = sum(sum(abs(msk_sampled(:,:,1) - msk_sampled_vc(:,:,vc))));
%     
%     for rz = 0:Rz-1
%         for ry = 0:Ry-1
%             sum_now = sum(sum(abs(msk_sampled(:,:,1) - circshift( msk_sampled_vc(:,:,vc), [-rz, -ry] ))));
%             
%             if sum_now < sum_best
%                 Delz_vc(vc) = rz;
%                 Dely_vc(vc) = ry;
%                 sum_best = sum_now;
%             end
%         end
%     end
% end
% 
% 
% DelZ = cat(2, Delz_vc*0, Delz_vc);
% DelY = cat(2, Dely_vc*0, Dely_vc);
% 
% % k-space offset of each phase-cycle:
% [DelZ(1:num_chan:end).', DelY(1:num_chan:end).']
% 
% 
% tic
% for n = 1:num_iter
%     disp(['Iter: ', num2str(n)])
%     
%     if n == 1
%         if use_sparse == 1
%             % initial sparse grappa recon        
%             [Img_grappa_DvX, ~, ~] = grappa_gfactor_2d_jvc2( kspace_sampled .* Fdx, kspace_acs .* Fdx, Rz, Ry, num_acs, kernel_size, lambda_tik, 1 );
%             
%             [Img_grappa_DhX, mask, mask_acs] = grappa_gfactor_2d_jvc2( kspace_sampled .* Fdy, kspace_acs .* Fdy, Rz, Ry, num_acs, kernel_size, lambda_tik, 1 );
%                       
%             Img_grappa = ifft2call( (conj(Fdx) .* fft2call(Img_grappa_DvX) + conj(Fdy) .* fft2call(Img_grappa_DhX)) ./ (abs(Fdx).^2 + abs(Fdy).^2 + eps) .* (1-W) + Xhatp );              
% 
%             rmse_sgrappa = rmse( rsos(Img_grappa,3), img_R1 )
% 
%             mosaic( rsos(Img_grappa,3), 1, 1, fig_num, ['s Grappa: ', num2str(rmse_sgrappa)], [0,1e-3], 180 ),  
%         else
%             % initial vc grappa recon
%             [Img_vcgrappa, mask, mask_acs] = grappa_gfactor_2d_jvc2( Kspace_sampled_vc, Kspace_acs_vc, Rz, Ry, num_acs, kernel_size, lambda_tik, 1, DelZ, DelY );
% 
%             rmse_vcgrappa = rmse( rsos(Img_vcgrappa(:,:,1:num_chan),3), img_R1 )
% 
%             Img_grappa = Img_vcgrappa(:,:,1:num_chan);
% 
%             mosaic( rsos(Img_vcgrappa(:,:,1:num_chan),3), 1, 1, fig_num, ['vc Grappa: ', num2str(rmse_vcgrappa)], [0,1e-3], 180 ),  
%         end    
%     else
%         % joint virtual coil grappa recon
%         Kspace_grappa_vc = fft2c( cat(3, Img_grappa, conj(Img_grappa) ) );
% 
%         Kspace_acs_vc = zeross([N, num_chan*2]);
% 
%         Kspace_acs_vc(1+end/2-num_acs_vc(1)/2:end/2+num_acs_vc(1)/2 + 1, 1+end/2-num_acs_vc(2)/2:end/2+num_acs_vc(2)/2 + 1, :) = ...
%             Kspace_grappa_vc(1+end/2-num_acs_vc(1)/2:end/2+num_acs_vc(1)/2 + 1, 1+end/2-num_acs_vc(2)/2:end/2+num_acs_vc(2)/2 + 1, :);
% 
%         Im_vcgrappa = grappa_gfactor_2d_jvc2( Kspace_sampled_vc, Kspace_acs_vc, Rz, Ry, num_acs_vc, kernel_size_vc, lambda_vc, 0, DelZ, DelY );
%         
%         % substitute sampled & acs data
%         Kspace_vcgrappa = fft2c(Im_vcgrappa(:,:,1:num_chan));
% 
%         Img_grappa = ifft2c( Kspace_vcgrappa .* (~mask & ~mask_acs) + kspace_sampled .* (~mask_acs) + kspace_acs );
%         
%         Img_Grappa_vc = reshape(Img_grappa, [N, num_chan]);
% 
%     
%         rmse_vcgrappa = rmse( rsos(Img_Grappa_vc,3), img_R1 )
%     
%         mosaic( rsos(Img_Grappa_vc, 3), 1, 1, fig_num+1, ['vc Grappa: ', num2str(rmse_vcgrappa)], [0,1e-3], 180 ),  
%     end
% end
% toc
% 
% 
% %--------------------------------------------------------------------------
% %% joint LORAKS    
% %--------------------------------------------------------------------------
% 
% addpath /autofs/cluster/kawin/berkin/Matlab_Code_New/Joint_LORAKS/JAC-LORAKS_fast_FFT/Joint_Loraks_Toolbox
% 
% 
% r_S = 350;       % rank constraint
%  
% fig_num = 13;
% R = 2;           % local k-space radius
% 
% pcg_tol = 1e-6;  % pcg tolerance to terminate
% pcg_iter = 100;  % pcg max iterations
% 
% 
% Mask_acs = zeross([N, num_chan]);
% Mask_acs(1+end/2-num_acs(1)/2:end/2+num_acs(1)/2 + 1, 1+end/2-num_acs(2)/2:end/2+num_acs(2)/2 + 1, :) = 1;
% 
% Mask_sampled = zeross([N, num_chan]);
% Mask_sampled(1:Rz:end, 1:Ry:end, :) = 1;
% 
% mask = reshape(Mask_acs + Mask_sampled, N(1), N(2), num_chan);
% mask = mask>0;
% 
% % undersampled data
% Kspace_sampled = fft2c2( IMG ) .* mask;
% 
% 
% kData = reshape(Kspace_sampled,N(1),N(2),[]);
% kMask = reshape(mask,N(1),N(2),[]);
% 
% % virtual coils
% kDatavirt = flipdim(flipdim(conj(kData),1),2);
% kMaskvirt = flipdim(flipdim(kMask,1),2);
% 
% 
% % combined actual and virtual coils
% kData = reshape([kData(:);kDatavirt(:)],N(1),N(2),[]);
% kMask = reshape([kMask(:);kMaskvirt(:)],N(1),N(2),[]);
% 
% N1 = size(kData,1);
% N2 = size(kData,2);
% Nc = size(kData,3);
% 
% 
% % Find loraks annhilating filters 
% Nic = calib_loraks( kData, kMask, R, r_S );
% 
% 
% S.type = '()';
% S.subs{:} = find(~vect(ifftshift(kMask)));
% 
% phi = @(x) subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(subsasgn(zeros(N1,N2,Nc),S,x)))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);
% b = -subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(reshape(ifftshift(kData),[N1,N2,Nc])))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);
% 
% 
% % PCG recon
% tic
%     [z, flag, res, iter, resvec] = pcg(@(x)  phi(x), b, pcg_tol, pcg_iter);
% toc
% 
% A = @(x) fftshift(ifftshift(kData) + subsasgn(zeros(N1,N2,Nc),S,x));
% k = A(z);
% 
% 
% Img_Loraks = reshape(ifft2c(reshape(k(1:N1*N2*Nc/2),[N1,N2,Nc/2])), [N, num_chan]);
% 
% 
% rmse_loraks = rmse( rsos(Img_Loraks, 3), rsos(img_R1,3) )
%         
% mosaic( rsos(Img_Loraks, 3), 1, 1, fig_num, ['Loraks: ', num2str(rmse_loraks)], [0,1e-3], 180 ),  
 

