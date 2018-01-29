%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  @inproceedings{zhang2017learning,
%    title={Learning Deep CNN Denoiser Prior for Image Restoration},
%    author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%    pages={3929--3938},
%    year={2017},
%  }
%   mosaic funtion
%
%    Input
%     - rgb        :  full RGB image
%     - pattern    : mosaic pattern
%     - noiselevel : noise level of Gaussian noise

%           pattern = 'grbg' % default
%            G R ..
%            B G ..
%            : :
%           pattern = 'rggb'
%            R G ..
%            G B ..
%            : :
%           pattern = 'gbrg'
%            G B ..
%            R G ..
%            : :
%           pattern = 'bggr'
%            B G ..
%            G R ..
%            : :
%
%    Output
%     - mosaic :  mosaiced image
%     - mask   :  binaly mask (3D data : height*width*RGB)
%     - B      :  (2D data : height*width)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [B, mosaic, mask] = mosaic_bayer(rgb, pattern, noiselevel)

num = zeros(size(pattern));
p = find((pattern == 'r') + (pattern == 'R'));
num(p) = 1;
p = find((pattern == 'g') + (pattern == 'G'));
num(p) = 2;
p = find((pattern == 'b') + (pattern == 'B'));
num(p) = 3;

mosaic = zeros(size(rgb, 1), size(rgb, 2), 3);
mask   = zeros(size(rgb, 1), size(rgb, 2), 3);
B      = zeros(size(rgb, 1), size(rgb, 2));


rows1 = 1:2:size(rgb, 1);
rows2 = 2:2:size(rgb, 1);
cols1 = 1:2:size(rgb, 2);
cols2 = 2:2:size(rgb, 2);

B(rows1, cols1) = rgb(rows1, cols1, num(1));
B(rows1, cols2) = rgb(rows1, cols2, num(2));
B(rows2, cols1) = rgb(rows2, cols1, num(3));
B(rows2, cols2) = rgb(rows2, cols2, num(4));

randn('seed',0);
B = B + noiselevel/255*randn(size(B));

mask(rows1, cols1, num(1)) = 1;
mask(rows1, cols2, num(2)) = 1;
mask(rows2, cols1, num(3)) = 1;
mask(rows2, cols2, num(4)) = 1;


mosaic(rows1, cols1, num(1)) = B(rows1, cols1);
mosaic(rows1, cols2, num(2)) = B(rows1, cols2);
mosaic(rows2, cols1, num(3)) = B(rows2, cols1);
mosaic(rows2, cols2, num(4)) = B(rows2, cols2);

% mosaic(:,:,1) = rgb(:,:,1) .* mask(:,:,1);
% mosaic(:,:,2) = rgb(:,:,2) .* mask(:,:,2);
% mosaic(:,:,3) = rgb(:,:,3) .* mask(:,:,3);




end
