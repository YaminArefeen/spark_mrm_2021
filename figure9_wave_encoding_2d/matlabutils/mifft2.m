function [ res ] = mifft2( DATA )

%%%%%%%%%%%%%%%%%%%%%%% mifft2 %%%%%%%%%%%%%%%%%%%%
% made by JaeJin Cho            2016.12.01  
% 
% ifft2 operater (first and second dimensions)
% [ res ] = mifft2( DATA )
% DATA    : data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

res = mifft( DATA, 1 );
res = mifft( res,  2 );

end
