function res = fft2c(x)
fctr = size(x,1)*size(x,2);
res = zeros(size(x));

size_x = size(x);

x = reshape(x, size_x(1), size_x(2), []);

% tic
for ii=1:size(x,3)
    res(:,:,ii) = 1/sqrt(fctr)*fftshift(fft2(ifftshift(x(:,:,ii))));
end
% toc

res = reshape(res, size_x);

end





