%Implementation of the psuedo replice technique with GRAPPA in preparation for SPARK
%   -Estimate noise-covariance from edges of fully sampled coil-images where there should be just noise
%   -Undersample and perform base-line reconstruction for a particular acceleration 
%   -Generate and perform GRAPPA reconstruction on all instances of k-space + synthesized noise
%   -Pass grappa reconstructed k-spaces, noisy k-spaces, and base-line k-spaces to sprk

addpath matlabutils/

%% Load the data, callibrate coil-sensitivity maps, set parameters, etc...
load('data/img_grappa_32chan.mat');
img_coils       = permute(IMG,[2,1,3]);
[M,N,C]         = size(img_coils);

%-Monte-carlo parameters
nmc     = 100;       %'number' of monte-carlo iterations

%-GRAPPA parameters
Rx      = 6;
Ry      = 1;
acs     = [30,N-2];
kernel  = [3,3];
lambda  = 0;
subs    = 0;

%-Generate undersampling mask
mask = zeros(M,N);
mask(1:Rx:end,1:Ry:end) = 1;

%-Generating k-space fully sampled
kspace_orig     = mfft2(img_coils);

%-Generating coils
fprintf("Callibrating maps... ");
coils = squeeze(bart('ecalib -m 1',reshape(kspace_orig,M,N,1,C)));

%-Generate undersampling mask
mask = zeros(M,N);
mask(1:Rx:end,1:Ry:end) = 1;

%% Estimating Noise from the coil imges + computing noise covariance
noise = reshape(img_coils(1:20,:,:),[],C);
Nk    = length(noise);

noise_covariance = zeros(C,C);

for ii = 1:C
    for jj = 1:C
        noise_covariance(ii,jj) = (1/(2*Nk)) * (noise(:,jj)'*noise(:,ii));
    end
end
[U,E] = eig(noise_covariance);
noise_whiten = (diag(diag(E).^(-1/2))*U'*(noise.')).';
noise_covariance_root = U * diag(diag(E).^(1/2)) * U';

%% Undersample and perform base-line reconstruction
kspace = kspace_orig .* mask; %under-sampled baseline k-space
cc     = @(x,coils) sum(conj(coils).*x,3)./(eps + sum(coils.*conj(coils),3)); %coil-combination

baseline_coils = grappa_gfactor_2d_jvc2(kspace,kspace_orig,Rx,Ry,acs,kernel,lambda,subs);
baseline = cc(baseline_coils,coils);

%% Generate and perform reconstructions for all nmc noise instances
kspace_grappa_noisy = zeros(M,N,C,nmc);
kspace_noisy        = zeros(M,N,C,nmc);
kspace_noisy_full   = zeros(M,N,C,nmc);  %To be used for ground truth later
replicas            = zeros(M,N,nmc);
for nn = 1:nmc
    fprintf("Iteration %d/%d\n",nn,nmc);
    tic
    noise_uncorrelated_real = randn(N,M,C);
    noise_uncorrelated_imag = randn(N,M,C);
    
    noise_correlated_real = reshape(permute(noise_covariance_root*permute(reshape(noise_uncorrelated_real,M*N,C),[2,1]),[2,1]),M,N,C);
    noise_correlated_imag = reshape(permute(noise_covariance_root*permute(reshape(noise_uncorrelated_imag,M*N,C),[2,1]),[2,1]),M,N,C);
    
    noise_correlated = noise_correlated_real + 1i * noise_correlated_imag;
    
    cur_kspace = (kspace + noise_correlated).*mask;
    cur_recon  = grappa_gfactor_2d_jvc2(cur_kspace,kspace_orig,Rx,Ry,acs,kernel,lambda,subs);
    
    replicas(:,:,nn)              = cc(cur_recon,coils);
    kspace_grappa_noisy(:,:,:,nn) = mfft2(cur_recon);
    kspace_noisy_full(:,:,:,nn)   = kspace_orig + noise_correlated;
    kspace_noisy(:,:,:,nn)        = cur_kspace;
    toc
end

%% Saving for SPARK
writecfl(sprintf('forspark/kspace_orig_Rx%dRy%d',Rx,Ry),          kspace_orig);
writecfl(sprintf('forspark/kspace_noisy_Rx%dRy%d',Rx,Ry),         kspace_noisy);
writecfl(sprintf('forspark/kspace_grappa_noisy_Rx%dRy%d',Rx,Ry),  kspace_grappa_noisy);
writecfl(sprintf('forspark/kspace_noisy_full_Rx%dRy%d',Rx,Ry),    kspace_noisy_full);

forspark.Rx     = Rx;
forspark.Ry     = Ry;
forspark.acsx   = acs(1);
forspark.acsy   = acs(2);

forspark.baseline_coils = baseline_coils;
forspark.coils          = coils;

save(sprintf('forspark/forspark_Rx%dRy%d.mat',Rx,Ry),'forspark');