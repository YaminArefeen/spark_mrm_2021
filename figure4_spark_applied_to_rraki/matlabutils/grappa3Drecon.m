% Reconstruction the 3D kspace using Berkin's 3D Grappa function

%% Setting some parameters
%-Acceleration Factors
Rx = 1;
Ry = 2;
Rz = 1;

%-Acs Size
acsx = 64;
acsy = 32;
acsz = 32;

%-GRAPPA parameters
kernel_size = [3,3,3];
lambda_tik  = eps;
subs        = 0;
acs = [acsx acsy acsz]-2;
%Substract the two here b/c Berkin usually does this
%% Loading the dataset and reference
fprintf('Loading dataset and reference... '); tic;
kspace = readcfl('kspace');
ref    = readcfl('reference');
fprintf('Elapsed Time: %.2f (s)\n',toc);

%% Performing 3D grappa reconstruction
fprintf('Performing 3D GRAPPA reconstruction...\n'); tic;
imgCoils = grappa_gfactor_3d_jvc(kspace,ref,Ry,Rz,acs,kernel_size,lambda_tik,subs);
fprintf('Elapsed Time: %.2f (s)\n',toc);

%% Generate the 3D kspace and save
fprintf('Generating and saving full kspace... '); tic;
kspaceFull = mfft3(imgCoils);
writecfl('kspaceFull',kspaceFull)

fprintf('Elapsed Time: %.2f (s)\n',toc);