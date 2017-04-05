% gray image denoising

% @inproceedings{zhang2017learning,
%   title={Learning Deep CNN Denoiser Prior for Image Restoration},
%   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%   year={2017}
% }

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

% clear; clc;
addpath('utilities');
imageSets     = {'BSD68','Set12'}; %%% testing dataset
folderTest    = 'testsets';
folderModel   = 'models';
folderResult  = 'results';
taskTestCur   = 'Denoising';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

setTestCur    = imageSets{1};
imageSigmaS   = [15,25,50];
modelSigmaS   = [15,25,50];
showResult    = 1;
saveResult    = 0;
useGPU        = 1;
pauseTime     = 1;

%%% folder to store results
folderResultCur = fullfile(folderResult, [taskTestCur,'_',setTestCur]);
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
folderTestCur = fullfile(folderTest,setTestCur);
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTestCur,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(length(modelSigmaS),length(filePaths));
SSIMs = zeros(length(modelSigmaS),length(filePaths));

load(fullfile(folderModel,'modelgray.mat'));

for i = 1:length(modelSigmaS)
    
    disp([i,length(modelSigmaS)]);
    net = loadmodel(modelSigmaS(i),CNNdenoiser);
    net = vl_simplenn_tidy(net);
    % for i = 1:size(net.layers,2)
    %     net.layers{i}.precious = 1;
    % end
    %%% move to gpu
    if useGPU
        net = vl_simplenn_move(net, 'gpu');
    end
    
    for j = 1:length(filePaths)
        
        %%% read images
        label = imread(fullfile(folderTestCur,filePaths(j).name));
        [~,imageName,extCur] = fileparts(filePaths(j).name);
        label = im2double(label);
        randn('seed',0);
        input = single(label + imageSigmaS(i)/255*randn(size(label)));
        
        %%% convert to GPU
        if useGPU
            input = gpuArray(input);
        end
        res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        output = input - res(end).x;
        
        %%% convert to CPU
        if useGPU
            output = gather(output);
            input  = gather(input);
        end
        
        %%% calculate PSNR and SSIM
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        if showResult
            imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
            title([filePaths(j).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            drawnow;
            if saveResult
                imwrite(im2uint8(output),fullfile(folderResultCur,[imageName,'_',num2str(imageSigmaS(i)),'_',num2str(modelSigmaS(i)),'_',num2str(PSNRCur,'%2.2f'),'.png']));
            end
            pause(pauseTime)
        end
        PSNRs(i,j) = PSNRCur;
        SSIMs(i,j) = SSIMCur;
        
    end
end

%%% save PSNR and SSIM metrics
save(fullfile(folderResultCur,['PSNR_',taskTestCur,'_',setTestCur,'.mat']),'PSNRs')
save(fullfile(folderResultCur,['SSIM_',taskTestCur,'_',setTestCur,'.mat']),'SSIMs')

disp([mean(PSNRs,2),mean(SSIMs,2)]);


