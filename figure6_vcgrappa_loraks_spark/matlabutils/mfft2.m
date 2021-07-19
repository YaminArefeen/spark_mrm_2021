function [ res ] = mfft2( DATA )
%%%%%%%%%%%%%%%%%%%%%%% mfft2 %%%%%%%%%%%%%%%%%%%%
% made by JaeJin Cho            2016.12.01  
% 
% fft2 operater (first and second dimensions)
% [ res ] = mfft2( DATA )
% DATA    : data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


res = mfft( DATA, 1 );
res = mfft( res,  2 );

end

