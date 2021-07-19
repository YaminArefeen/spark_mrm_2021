function res = mfft3(DATA)
res = mfft(mfft(mfft(DATA,1),2),3);
end
