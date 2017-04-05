function [LR] = imresize_down(im, scale, type, sigma)

if nargin ==3 && strcmp(type,'Gaussian')
    sigma = 1.6;
end

if strcmp(type,'Gaussian') && fix(scale) == scale
    if mod(scale,2)==1
        kernelsize = ceil(sigma*3)*2+1;
        if scale==3 && sigma == 1.6
            kernelsize = 7;
        end
        kernel  = fspecial('gaussian',kernelsize,sigma);
        blur_HR = imfilter(im,kernel,'replicate');
        
        if isa(blur_HR, 'gpuArray')
            LR = blur_HR(scale-1:scale:end-1,scale-1:scale:end-1,:);
        else
            LR      = imresize(blur_HR, 1/scale, 'nearest');
        end
        
        
        % LR      = im2uint8(LR);
    elseif mod(scale,2)==0
        kernelsize = ceil(sigma*3)*2+2;
        kernel     = fspecial('gaussian',kernelsize,sigma);
        blur_HR    = imfilter(im, kernel,'replicate');
        LR= blur_HR(scale/2:scale:end-scale/2,scale/2:scale:end-scale/2,:);
        % LR         = im2uint8(LR);
    end
else
    LR = imresize(im, 1/scale, type);
end











