function [ res ] = mifft( x, dim )
%%%%%%%%%%%%%%%%%%%%%%%% mifft %%%%%%%%%%%%%%%%%%%%
% made by JaeJin Cho            2016.12.01  
% 
% ifft operater
% [ res ] = mifft( DATA, dim )
% DATA    : data
% dim     : dimension number
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% res = ifftshift(ifft(ifftshift(DATA,dim),[],dim),dim) ;% .* (size(DATA,dim));

res = sqrt(size(x,dim))*fftshift(ifft(ifftshift(x,dim),[],dim),dim);

end

