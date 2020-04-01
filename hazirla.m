%%
clear,clc

%%
imds = imageDatastore('C:\Users\FIRAT-PC\Desktop\CROP\BRN_ELEME');
sayi = length(imds.Files);
for i = 1:sayi
    A = readimage(imds,i);
    filename = char(imds.Files(i));
    imwrite(A,filename);
end

%%
imds = imageDatastore('C:\Users\FIRAT-PC\Desktop\CROP\BINARY_ELEME');
sayi = length(imds.Files);
for i = 1:sayi
    A = readimage(imds,i);
    B = uint8(A);
    filename = char(imds.Files(i));
    imwrite(B,filename);
end