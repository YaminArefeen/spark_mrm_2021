function [ Nic ] = calib_loraks( kData, kMask, R, r_S )
%CALIB_LORAKS Summary of this function goes here
%   Detailed explanation goes here




N1 = size(kData,1);
N2 = size(kData,2);
Nc = size(kData,3);

[P_C, Ph_C, ~, ~, ~, ~, cc, ~, ~, sizeC, ~, ~] = generate_LORAKS_operators(N1, N2, R);


% Define P-LORAKS Operators
st = ['P_CC=@(x) [ '];

for i = 1:Nc
    st = [st, 'P_C(x([1:N1*N2]+' num2str(i-1) '*N1*N2));'];
end

st = [st,'];'];
eval(st);


st = ['Ph_CC=@(x) [ '];

for i = 1:Nc
    st = [st, 'Ph_C(x([1:sizeC(1)]+' num2str(i-1) '*sizeC(1),:));'];
end

st = [st(1:end-1),'];'];
eval(st);




%--------------------------------------------------------------------------
% Generate Subspaces
%--------------------------------------------------------------------------


disp('Estimating subspaces');


C = P_CC(kData(:));

Csamples = P_CC(kMask(:));
indC = find(sum(Csamples,1)==size(Csamples,1));

Csmall = C(:,indC);

[u,t1,~] = svd(Csmall,'econ');

n = u(:,r_S+1:end);
u = u(:,1:r_S);




%--------------------------------------------------------------------------
% FFT-based matrix multiplication
% Find image domain representations of annhilating filters
%--------------------------------------------------------------------------


disp('FFT-based precomputations');


ncc = n';                   % annhilating filters 
fsz = size(ncc,2)/Nc;       % each channel annihilating filter size
numflt = size(ncc,1);       % number of annihilating filters
N1_ac = numel(-R:R);        % annihilating filter patch size
N2_ac = numel(-R:R);    


[in1,in2] = meshgrid(-R:R,-R:R);    
idx = find(in1.^2+in2.^2<=R^2);
in1 = in1(idx)';
in2 = in2(idx)';

i = floor(N1_ac/2)+1;
j = floor(N2_ac/2)+1;
ind = sub2ind([N1_ac, N2_ac],i+in1,j+in2);

N1_ac = 2*N1_ac-1;
N2_ac = 2*N2_ac-1;

x_ac_idx = (floor(N1/2)+1 - floor(N1_ac/2)) : (floor(N1/2)+1 + floor(N1_ac/2) - ~rem(N1_ac,2));
y_ac_idx = (floor(N2/2)+1 - floor(N2_ac/2)) : (floor(N2/2)+1 + floor(N2_ac/2) - ~rem(N2_ac,2));
zp_patch = zeros(N1, N2);   %zeropadded annhilating filter

Nic = zeros(Nc*N1*N2,Nc);

filtfilt = zeros((2*R+1)*(2*R+1),Nc,numflt);
tmptmp = reshape(permute(ncc,[2,1]),[fsz,Nc,numflt]);
filtfilt(ind,:,:) = tmptmp;
filtfilt = reshape(filtfilt,(2*R+1),(2*R+1),Nc,numflt);

cfilt = conj(filtfilt);
ccfilt = zeros(N1_ac,N1_ac,Nc,numflt);
ccfilt(2*R+1+[-R:R],2*R+1+[-R:R],:,:) = cfilt;

 

ffilt = zeros(size(filtfilt));

for a = 1:size(filtfilt,4)
    for b = 1:size(filtfilt,3)
        ffilt(:,:,b,a) = rot90(filtfilt(:,:,b,a), 2);
    end
end


fffilt = zeros(N1_ac,N1_ac,Nc,numflt);
fffilt(2*R+1+[-R:R],2*R+1+[-R:R],:,:) = ffilt;

ccfilt = single(fft2(ifftshift(ccfilt)));
fffilt = single(fft2(ifftshift(fffilt)));

for p = 1:Nc
    for l = 1:Nc
        patch = fftshift(ifft2(sum(ccfilt(:,:,p,:).*fffilt(:,:,l,:),4)));           % it's possible to remove these fftshifts at each iteration
        zp_patch(x_ac_idx, y_ac_idx,:) = patch;
        tmp(:,l) = vect(ifft2(ifftshift(zp_patch)))*(N1*N2)*sqrt(N1*N2);            % it's possible to remove these fftshifts at each iteration
    end
    Nic((p-1)*N1*N2+[1:N1*N2],:) = single(tmp);
end

Nic = reshape(permute(reshape(Nic,[N1*N2,Nc,Nc]),[1,3,2]),[N1*N2*Nc,Nc]);
 



end

