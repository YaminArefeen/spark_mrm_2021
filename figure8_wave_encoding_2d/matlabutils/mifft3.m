function res = mifft3(DATA)
res = mifft(mifft(mifft(DATA,1),2),3);
end
