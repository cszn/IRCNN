function [ BF,BCF,B2F,y,Bpad] = HXconv( x,B,conv )
%************************************************
% Author: Ningning Zhao (University of Toulouse)
% Date: 2015/03/28
% Note: if nargin==2, B must be the sfft2(psf)
%       if nargin==3, B must be the psf (RF form)
%      -> x is the RF signal or isreal(x)=1
%      -> B is shift invariant and 
%         circular boundary is considered
%************************************************

[m,n] = size(x);
[m0,n0]=size(B);
% Bpad=padarray(B,floor([m-m0+1,n-n0+1]/2),'pre');
% Bpad=padarray(Bpad,round([m-m0-1,n-n0-1]/2),'post');
% Bpad=fftshift(Bpad);
% BF = fft2(Bpad);

BF = psf2otf(B,[m,n]);

BCF = conj(BF);
B2F = abs(BF).^2;

if nargin ==2
    return;
elseif nargin==3
    switch conv

        case 'Hx'
            y = real(ifft2(BF.*fft2(x)));
        case 'HTx'
            y = real(ifft2(BCF.*fft2(x)));
        case 'HTHx'
            y = real(ifft2(B2F.*fft2(x)));
    end

end



