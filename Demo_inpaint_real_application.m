%==========================================================================
% This is the testing code of IRCNN for image inpainting.
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
imageSets    = {'Inpaint_set2'}; % testing dataset
setTest      = imageSets(1); % select the dataset

useGPU       = 1;

folderTest   = 'testsets';
folderResult = 'results';
folderModel  = 'models';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end
setTestCur = cell2mat(setTest(1));
disp('--------------------------------------------');
disp(['----',setTestCur,'-----Image Inpainting-----']);
disp('--------------------------------------------');
folderTestCur = fullfile(folderTest,setTestCur);

% folder to store results
folderResultCur = fullfile(folderResult, ['Inpaint_',setTestCur]);
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end

%% read original image 'Iori' and its mask

Iname       = 'new';  % Isigma = 0.5/255; Msigma = 5; window      = 10;  %Images from http://www.visinf.tu-darmstadt.de/vi_research/code/foe.en.jsp
Iname       = '3ch';  % Isigma = 0.5/255; Msigma = 5; window      = 30;  %
%  [1] S. Roth and M. J. Black, ¡°Fields of experts: A framework for learning image priors,¡± CVPR, vol. 2, San Diego, California, Jun. 2005, pp. 860¨C867.

%  window, important!
window      = 10;  % default. For '3ch.png',  window      = 30; 
if strcmp(Iname,'3ch') == 1 
   window      = 30;
end

Iori   = im2single(imread(fullfile(folderTestCur,[Iname,'.png'])));

[a,b,c] = size(Iori);

% load mask

mask  = logical(imread(fullfile(folderTestCur,[Iname,'_mask.png'])));
mask  = 1- mask;

% rand('seed',0);
% mask = rand(a,b)>=0.8;
mask = repmat(mask,[1,1,c]);

% generate input
y = Iori.*mask;


%% parameter setting in HQS (tune the following parameters to obtain the best results)
% -------------------important!------------------
% Parameter settings of IRCNN
% (1) image noise level: Isigma
Isigma = 0.5/255; % ****** from interval [1/255, 20/255] ******; e.g., 1/255, 2.55/255, 7/255, 11/255
% (2) noise level of the last denoiser: Msigma
Msigma = 5; % ****** from {1 3 5 7 9 11 13 15} ******
%--------------------------------------------------------


%% load denoisers
if c==1
    load(fullfile(folderModel,'modelgray.mat'));
elseif c==3
    load(fullfile(folderModel,'modelcolor.mat'));
end

%% default parameter setting in HQS
totalIter   = 30; % default 30
lamda       = (Isigma^2)/3; % default 3, ****** from {1 2 3 4} ******
modelSigma1 = 49; % default 49
modelSigmaS = logspace(log10(modelSigma1),log10(Msigma),totalIter);
rho         = Isigma^2/((modelSigma1/255)^2);

ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];


z           = shepard_initialize(y, mask, window);

if useGPU
    z       = gpuArray(z);
    y       = gpuArray(y);
end


for itern = 1:totalIter
    
    % step 1
    rho = lamda*255^2/(modelSigmaS(itern)^2);
    z   = (y+rho*z)./(mask+rho);
    
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

imshow(cat(2,im2uint8(Iori),output));

imwrite(output,fullfile(folderResultCur,[Iname,'_ircnn.png']));













