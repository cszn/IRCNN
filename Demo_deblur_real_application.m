%==========================================================================
% This is the testing code of IRCNN for image deblurring with estimated
% kernel by other blind deblurring methods.
%
% There are two important parameters to tune:
% (1) image noise level of blurred image: Isigma and
% (2) noise level of the last denoiser: Msigma.
%
% @inproceedings{zhang2017learning,
%   title={Learning Deep CNN Denoiser Prior for Image Restoration},
%   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%   year={2017}
% }
%
% If you have any question, please feel free to contact with <Kai Zhang (cskaizhang@gmail.com)>.
%
%
% by Kai Zhang (1/2018)
%==========================================================================

clear; clc;

addpath('utilities');
imageSets    = {'Deblur_set1','Deblur_set2','Deblur_set3'}; % testing dataset
setTest      = imageSets(2); % select the dataset

useGPU       = 1;

folderTest   = 'testsets';
folderResult = 'results';
folderModel  = 'models';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end
setTestCur = cell2mat(setTest(1));
disp('--------------------------------------------');
disp(['----',setTestCur,'-----Image Debluring-----']);
disp('--------------------------------------------');
folderTestCur = fullfile(folderTest,setTestCur);

% folder to store results
folderResultCur = fullfile(folderResult, ['Deblur_',setTestCur]);
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end

%% read blurred image and its estimated kernel
% blurred image
Iname = 'im01_ker01';
y  = im2single(imread(fullfile(folderTestCur,[Iname,'.png'])));
% estimated kernel
%k  = imread(fullfile(folderTestCur,[Iname,'_kernel.png']));
k  = imread(fullfile(folderTestCur,[Iname,'_out_kernel.png']));

if size(k,3)==3
    k = rgb2gray(k);
end
k  = im2single(k);
k  = k./(sum(k(:)));


%% -------------------important!------------------
% Parameter settings of IRCNN
% (1) image noise level of blurred image: Isigma
Isigma = 5/255; % ****** from interval [1/255, 20/255] ******; e.g., 1/255, 2.55/255, 7/255, 11/255
% (2) noise level of the last denoiser: Msigma
Msigma = 5; % ****** from {1 3 5 7 9 11 13 15} ******
%--------------------------------------------------------


[a1,b1,~] = size(y);

%% handle boundary
boundary_handle = 'case2';
switch boundary_handle
    case {'case1'} % option (1), edgetaper to better handle circular boundary conditions, (matlab2015b)
        % k(k==0) = 1e-10; % uncomment this for matlab 2016--2018?
        ks = floor((size(k) - 1)/2);
        y = padarray(y, ks, 'replicate', 'both');
        for a=1:4
            y = edgetaper(y, k);
        end
    case {'case2'} % option (2)
        H = size(y,1);    W = size(y,2);
        y = wrap_boundary_liu(y, opt_fft_size([H W]+size(k)-1));
end


[w,h,c]  = size(y);
V = psf2otf(k,[w,h]);
denominator = abs(V).^2;

if c>1
    denominator = repmat(denominator,[1,1,c]);
    V = repmat(V,[1,1,c]);
end
upperleft   = conj(V).*fft2(y);

% load denoisers
if c==1
    load(fullfile(folderModel,'modelgray.mat'));
elseif c==3
    load(fullfile(folderModel,'modelcolor.mat'));
end

totalIter   = 30; % default 30
lamda       = (Isigma^2)/3; % default 3, ****** from {1 2 3 4} ******
modelSigma1 = 49; % default 49
modelSigmaS = logspace(log10(modelSigma1),log10(Msigma),totalIter);
rho         = Isigma^2/((modelSigma1/255)^2);

ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];

z           = single(y);
if useGPU
    z           = gpuArray(z);
    upperleft   = gpuArray(upperleft);
    denominator = gpuArray(denominator);
end

for itern = 1:totalIter
    % step 1
    rho = lamda*255^2/(modelSigmaS(itern)^2);
    z = real(ifft2((upperleft + rho*fft2(z))./(denominator + rho)));
    if ns(itern+1)~=ns(itern)
        [net] = loadmodel(modelSigmaS(itern),CNNdenoiser);
        net = vl_simplenn_tidy(net);
        if useGPU
            net = vl_simplenn_move(net, 'gpu');
        end
    end
    % step 2
    res = vl_simplenn(net, z,[],[],'conserveMemory',true,'mode','test');
    residual = res(end).x;
    z = z - residual;
    
    %     imshow(z)
    %     title(int2str(itern))
    %     drawnow;
end


if useGPU
    output = im2uint8(gather(z));
end

switch boundary_handle
    case {'case1'} % option (1)
        output = center_crop(output,a1,b1);
        y = center_crop(y,a1,b1);
    case {'case2'} % option (2)
        output = output(1:a1,1:b1,:);
        y      = y(1:a1,1:b1,:);
end

imshow(cat(2,im2uint8(y),output));
imwrite(output,fullfile(folderResultCur,[Iname,'_ircnn_Isigma_',int2str(Isigma*255),'_Msigma_',int2str(Msigma),'.png']));







