%==========================================================================
% This is the testing code of IRCNN for color image demosaiking.
%
%  @inproceedings{zhang2017learning,
%    title={Learning Deep CNN Denoiser Prior for Image Restoration},
%    author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%    pages={3929--3938},
%    year={2017},
%  }
%
% If you have any question, please feel free to contact with <Kai Zhang (cskaizhang@gmail.com)>.
%
% -----------McMaster18--------
% --Set18----Color Demosaiking-
% -----------------------------
% 01.tif  --  30.26dB  --  0.93
% 02.tif  --  35.26dB  --  0.94
% 03.tif  --  34.69dB  --  0.97
% 04.tif  --  38.37dB  --  0.99
% 05.tif  --  35.09dB  --  0.95
% 06.tif  --  39.19dB  --  0.97
% 07.tif  --  39.66dB  --  0.98
% 08.tif  --  39.44dB  --  0.97
% 09.tif  --  38.64dB  --  0.96
% 10.tif  --  39.51dB  --  0.97
% 11.tif  --  40.46dB  --  0.97
% 12.tif  --  38.86dB  --  0.96
% 13.tif  --  40.71dB  --  0.95
% 14.tif  --  38.99dB  --  0.96
% 15.tif  --  39.47dB  --  0.96
% 16.tif  --  34.39dB  --  0.95
% 17.tif  --  34.79dB  --  0.96
% 18.tif  --  36.21dB  --  0.96
% Average PSNR and SSIM
%    37.4447dB    0.9614

% ----------Kodak24-----------------
% ----Set24-----Color Demosaiking---
% ----------------------------------
% kodim01.png  --  40.30dB  --  0.99
% kodim02.png  --  39.79dB  --  0.97
% kodim03.png  --  43.63dB  --  0.98
% kodim04.png  --  41.21dB  --  0.98
% kodim05.png  --  39.24dB  --  0.99
% kodim06.png  --  40.54dB  --  0.99
% kodim07.png  --  43.26dB  --  0.99
% kodim08.png  --  37.70dB  --  0.98
% kodim09.png  --  42.07dB  --  0.97
% kodim10.png  --  42.03dB  --  0.98
% kodim11.png  --  40.55dB  --  0.98
% kodim12.png  --  42.96dB  --  0.98
% kodim13.png  --  36.94dB  --  0.98
% kodim14.png  --  38.98dB  --  0.98
% kodim15.png  --  40.59dB  --  0.97
% kodim16.png  --  43.05dB  --  0.99
% kodim17.png  --  41.38dB  --  0.98
% kodim18.png  --  38.15dB  --  0.98
% kodim19.png  --  40.63dB  --  0.98
% kodim20.png  --  41.30dB  --  0.98
% kodim21.png  --  40.27dB  --  0.98
% kodim22.png  --  39.14dB  --  0.98
% kodim23.png  --  43.05dB  --  0.98
% kodim24.png  --  36.22dB  --  0.98
% Average PSNR and SSIM
%    40.5409dB    0.9806
%
% by Kai Zhang (1/2018)
%==========================================================================

clear; clc;

addpath('utilities');
imageSets    = {'Set18','Set24'}; % testing dataset
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
disp(['----',setTestCur,'--Color Image Demosaiking--']);
disp('--------------------------------------------');
folderTestCur = fullfile(folderTest,setTestCur);

% folder to store results
folderResultCur = fullfile(folderResult, ['Demosaik_',setTestCur]);
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end


%% Noise level 
noiselevel = 0; % default; noiselevel = 10; Isigma     = 10/255; Msigma     = 8;


%% parameter setting in HQS (tune the following parameters to obtain the best results)
%% -------------------important!------------------
% Parameter settings of IRCNN
% (1) image noise level: Isigma
Isigma     = 0.5/255; % default 0.5/255 for noise-free image, ****** from interval [1/255, 20/255] ******; e.g., 1/255, 2.55/255, 7/255, 11/255
% (2) noise level of the last denoiser: Msigma
Msigma     = 2; % default 2 for noise-free image, ****** from {1 2 3 4 5 7 9 11 13 15} ******
%--------------------------------------------------------

%% load denoisers
load(fullfile(folderModel,'modelcolor.mat'));

%% default parameter setting in HQS
totalIter   = 30; % default 30
lamda       = (Isigma^2)/3; % default 3, ****** from {1 2 3 4} ******
modelSigma1 = 49; % default 49
modelSigmaS = logspace(log10(modelSigma1),log10(Msigma),totalIter);
rho         = Isigma^2/((modelSigma1/255)^2);

ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];

ext                 =  {'*.jpg','*.png','*.bmp','*.tif'};
filepaths           =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
end

PSNRs = zeros(1,length(filepaths));
SSIMs = zeros(1,length(filepaths));

for i = 1 : length(filepaths)
    
    label  = imread(fullfile(folderTestCur,filepaths(i).name));
    [~, Iname, ext] = fileparts(filepaths(i).name);
    label = im2single(label);
    
    % generate mask
    [B, y, mask] = mosaic_bayer(label, 'grbg', noiselevel);
    y    = single(y);
    mask = single(mask);
    z = linearlcc(B, 0);
    z =  single(z);
    
    z0          = z;
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
        
        %         imshow(z)
        %         title(int2str(itern))
        %         drawnow;
    end
    
    if useGPU
        output = im2uint8(gather(z));
        y      = im2uint8(gather(y));
    end
    
    %output(mask==1) = y(mask==1);
    
    [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(im2uint8(label),output,10,10);
    
    PSNRs(i) = PSNR_Cur;
    SSIMs(i) = SSIM_Cur;
    
    imshow(cat(2,y,output));
    drawnow;
    pause(0.001);
    
    disp([filepaths(i).name,'  --  ', num2str(PSNR_Cur,'%2.2f'),'dB  --  ', num2str(SSIM_Cur,'%2.2f')]);
    %     imwrite(y,fullfile(folderResultCur,[Iname,'_mosaik.png']));
    %     imwrite(output,fullfile(folderResultCur,[Iname,'_ircnn.png']));
    
end

disp('Average PSNR and SSIM')
disp([mean(PSNRs),mean(SSIMs)]);

