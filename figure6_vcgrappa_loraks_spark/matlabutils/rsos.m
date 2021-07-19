function [ img_sos ] = rsos( img, chan_dim )
%RSOS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    chan_dim = 4;
end

img_sos = sum(abs(img).^2, chan_dim).^.5;


end

