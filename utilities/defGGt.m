function [G,Gt] = defGGt(h,K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Operators for super-resolution
% Stanley Chan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G  = @(x) fdown(x,h,K);
Gt = @(x) upf(x,h,K);
end

function y = fdown(x,h,K)
tmp = imfilter(x,h,'circular');
y = downsample2(tmp,K);
end

function y = upf(x,h,K)
tmp = upsample2(x,K);
y = imfilter(tmp,h,'circular');
end
