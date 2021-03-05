function res = fft2call(x)
fctr = size(x,1)*size(x,2);

X = fftshift(fft(ifftshift(x,1),[],1),1);

res = fftshift(fft(ifftshift(X,2),[],2),2) / sqrt(fctr);



