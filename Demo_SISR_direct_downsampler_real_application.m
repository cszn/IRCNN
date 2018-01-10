%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of IRCNN for image super-resolution where the latent HR image x is blurred and then downsampled to get the LR image y
% (y can be corrupted by additive Gaussian noise of level Isigma).
%
% The details of this degradation can be found by the following paper.
% [1] S. H. Chan, X. Wang, and O. A. Elgendy "Plug-and-Play ADMM for image restoration: Fixed point convergence and applications", IEEE Transactions on Computational Imaging, 2016.
%
% The objective function is given by min_x 1/(Isigma^2)||x*k_{direct downsampler with scale factor sf}-y||^2 + Phi(x)
%
%                  k  --  blur kernel, not limited to Gaussian blur
% direct downsampler  --  implemented by matlab function "downsample",
%                 sf  --  scale factor, 2,3,4,...
%             Isigma  --  estimated noise level of y, should be larger than the true one.
%
% @inproceedings{zhang2017learning,
%   title={Learning Deep CNN Denoiser Prior for Image Restoration},
%   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%   year={2017}
% }
%
%
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear; clc;

addpath('utilities');

imageSets   = {'SISR_set1','SISR_set2'}; % testing dataset

%%% setting
setTest     = imageSets(1); % select the dataset
showResult  = 1;
pauseTime   = 0;
useGPU      = 1; % 1 or 0, true or false

folderTest  = 'testsets';
folderResult= 'results';
taskTestCur = 'SISR';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end
setTestCur = cell2mat(setTest(1));
disp('--------------------------------------------');
disp(['----',setTestCur,'-----Super-Resolution-----']);
disp('--------------------------------------------');
folderTestCur = fullfile(folderTest,setTestCur);

%% read LR image and its estimated kernel under scale factor sf

%  ----- SISR_set1 ------
use_edgetaper = 0;              % use edgetaper to better handle circular boundary conditions! For SISR_set1, use_edgetaper = 0;
Iname = 'LR1_motion_x2';        %  Isigma      = 1/255; Msigma      = 1;
%Iname = 'LR2_gaussian_x2';     % Isigma      = 15/255; Msigma      = 7;
%Iname = 'LR3_noisy_x2';        % Isigma      = 12/255; Msigma      = 15;

%  ----- SISR_set2 ------ 
% use_edgetaper = 1;             % use edgetaper to better handle circular boundary conditions! For SISR_set2, use_edgetaper = 1;
% Iname = 'chip';                % sf = 2; Isigma      =  5/255; kernelsigma =  1;
% Iname = 'David_Hilbert';       % sf = 2; Isigma      = 15/255; kernelsigma = 0.8;
% Iname = 'Frog';                % sf = 2; Isigma      = 25/255; kernelsigma = 0.8; 


%% read LR image
LR    = imread(fullfile(folderTestCur,[Iname,'.png']));
c     = size(LR,3);

%% parameter setting in HQS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Important!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sf          = 2;      % scale factor {2,3,4,...}
Isigma      = 1/255; % default 1/255 for noise-free case. It should be larger than true noise level.
Isigma      = max(Isigma, 0.1/255);
Msigma      = 1;      % {1,3,5,7,9, ..., 15, ...}


% folder to store results
folderResultCur = fullfile(folderResult, ['SISR_direct_downsample_',setTestCur,'_x',num2str(sf)]);
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end

%% set blur kernel
% blur kernel k, not limited to Gaussian blur
% case 1
if exist(fullfile(folderTestCur,[Iname,'_kernel.png']),'file')
    k     = imread(fullfile(folderTestCur,[Iname,'_kernel.png']));
    if size(k,3)==3
        k = rgb2gray(k);
    end
    k    = im2single(k);
    k    = k./(sum(k(:)));
else
    % case 2
    kernelsigma = 0.8; % width (sigma) of the Gaussian blur kernel from [0.6 2.4], e.g., sf = 2, kernelsigma = 1; sf = 3, kernelsigma = 1.6; sf = 4, kernelsigma = 2;
    % k       = fspecial('gaussian',5,kernelsigma);
    k       = fspecial('gaussian',sf*2+5,kernelsigma);  % should be odd number.
    % k       = fspecial('gaussian',11,kernelsigma);
end


%% default parameter setting in HQS
totalIter   = 30;
modelSigmaS = logspace(log10(49),log10(Msigma),totalIter);
ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];
lamda       = (Isigma^2)/3; % default 3, ****** from {1 2 3 4} ******


folderModel = 'models';
% load denoisers
if c==1
    load(fullfile(folderModel,'modelgray.mat'));
elseif c==3
    load(fullfile(folderModel,'modelcolor.mat'));
end


HR_bic     = imresize(LR,sf,'bicubic');
[a1,b1,~]  =  size(HR_bic);
% input (single)
input      = im2single(HR_bic);


%% edgetaper to better handle circular boundary conditions
if use_edgetaper
    ks = sf*ceil(floor((size(k) - 1)/2)/sf);
    input = padarray(input, ks, 'replicate', 'both');
    for a=1:4
        input = edgetaper(input, k);
    end
    LR_edge      = downsample2(input, sf);  % downsampled
    LR = center_replace(LR_edge,im2single(LR));
end

%% prapare for step 1
y = im2single(LR);
[rows_in,cols_in,~] = size(y);
rows      = rows_in*sf;
cols      = cols_in*sf;
[G,Gt]    = defGGt(double(k),sf);
GGt       = constructGGt(k,sf,rows,cols);
if c == 3
    GGt   = cat(3,GGt,GGt,GGt); % R,G,B channels
end
Gty       = Gt(y);

if useGPU
    input = gpuArray(input);
    GGt   = gpuArray(GGt);
    Gty   = gpuArray(Gty);
end

output    = input;

%% main loop
tic;
for itern = 1:totalIter
    
    % step 1, closed-form solution, see Chan et al. [1] for details
    rho    = lamda*255^2/(modelSigmaS(itern)^2);
    rhs    = Gty + rho*output;
    output = (rhs - Gt(real(ifft2(fft2(G(rhs))./(GGt + rho)))))/rho;
    
    % load denoiser
    if ns(itern+1)~=ns(itern)
        net = loadmodel(modelSigmaS(itern),CNNdenoiser);
        net = vl_simplenn_tidy(net);
        if useGPU
            net = vl_simplenn_move(net, 'gpu');
        end
    end
    
    % step 2, perform denoising
    res    = vl_simplenn(net, output,[],[],'conserveMemory',true,'mode','test');
    im     = res(end).x;    % residual image
    output = output - im;
    % imshow(output)
    % drawnow;
    % pause(1)
end

if useGPU
    output = gather(output);
end
toc;

if use_edgetaper
    input  = center_crop(input,a1,b1);
    output = center_crop(output,a1,b1);
end

if showResult
    imshow(cat(2,input,output));
    drawnow;
    pause(pauseTime)
    imwrite(output,fullfile(folderResultCur,[Iname,'_ircnn_x',num2str(sf),'.png']));
end






