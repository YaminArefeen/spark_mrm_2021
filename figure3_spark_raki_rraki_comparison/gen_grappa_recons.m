%In this script, I want to generate GRAPPA reconstructions for a variety of ACS sizes and acceleration factors.  Thus,
%when performing ablation study on SPARK, I do not need to perform GRAPPA reconstruction's in python, rather, just need 
%to load the appropriate reconstructions
addpath 'matlabutils'

%% Setting the parameters for our reconstruction
accelerations = [5,6];           %-Acceleration in second dimension
acs_sizes     = [20,24,30,36,40]; %-Acs_size in second dimension

%-Grappa reconstruction parameters
kernel  = [3,3];
lambda  = 0;
subs    = 0;
Rx      = 1;

%-Loading the data
load('data/img_grappa_32chan.mat')
kspace = mfft2(IMG);
[M,N,C] = size(kspace);

acsx = M;

%% Running for loop over the different accelerations and acs_sizes
A = length(accelerations);
S = length(acs_sizes);

all_kspace_grappa = zeros(M,N,C,A,S);

ctr = 1;
for aa = 1:A
    for ss = 1:S    
        tic
        %-Setting parameters for this particular reconstruction
        Ry   = accelerations(aa);
        acsy = acs_sizes(ss);
        
        fprintf("Recon %d/%d || R = %d || acsy = %d\n",ctr,A*S,Ry,acsy); ctr = ctr + 1;
        
        kspace_und = zeros(M,N,C);
        kspace_und(1:Rx:end,1:Ry:end,:) = kspace(1:Rx:end,1:Ry:end,:);
        
        cur_recon = grappa_gfactor_2d_jvc2(kspace_und,kspace,Rx,Ry,[acsx,acsy]-2,kernel,lambda,subs);
        
        all_kspace_grappa(:,:,:,aa,ss) = mfft2(cur_recon);
        toc
    end
end

%% Save the different k-spaces
%-Saving the original, fully sampled k-space
save('grappa_recons/kspace_full.mat','kspace')

%-Saving reconstructed GRAPPA k-space for each acceleration and acs-size
for aa = 1:A
    for ss = 1:S
        kspace_grappa = all_kspace_grappa(:,:,:,aa,ss);
        save(sprintf('grappa_recons/kspace_grappa_Rx%dRy%dacsx%dacsy%d.mat',Rx,accelerations(aa),acsx,acs_sizes(ss)), ...
            'kspace_grappa');
    end
end