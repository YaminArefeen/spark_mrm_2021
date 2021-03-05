function res = ifft2call(x)
fctr = size(x,1)*size(x,2);
X = fftshift(ifft(ifftshift(x,1),[],1),1);
res = fftshift(ifft(ifftshift(X,2),[],2),2) * sqrt(fctr);


