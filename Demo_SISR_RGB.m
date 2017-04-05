% Single Image Super-Resolution (SISR)

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
imageSets   = {'Set5','Set14'}; % testing dataset

setTest     = imageSets([1]); % select the dataset
showResult  = 1;
pauseTime   = 1;
useGPU      = 1; % 1 or 0, true or false

folderTest  = 'testsets';
folderResult= 'results';
taskTestCur = 'SISR';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

kernelTypes = {'bicubic','Gaussian'};
kernelType_image  = kernelTypes{1};
kernelType_model  = kernelTypes{1};

scaleFactor = 3;
totalIter   = 30;
inIter      = 5;
alpha       = 1.75;
kernelsigma = 1.6; % ****** from [0.6 2.4] ******
modelSigmaS = logspace(log10(12*scaleFactor),log10(scaleFactor),totalIter);
ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];

folderModel = 'models';
load(fullfile(folderModel,'modelcolor.mat'));

for n_set = 1 : numel(setTest)
    %%% read images
    setTestCur = cell2mat(setTest(n_set));
    disp('--------------------------------------------');
    disp(['----',setTestCur,'-----Super-Resolution-----']);
    disp('--------------------------------------------');
    folderTestCur = fullfile(folderTest,setTestCur);
    ext                 =  {'*.jpg','*.png','*.bmp'};
    filepaths           =  [];
    for i = 1 : length(ext)
        filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
    end
    eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),' = zeros(length(filepaths),1);']);
    eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),' = zeros(length(filepaths),1);']);
    
    %%% folder to store results
    folderResultCur = fullfile(folderResult, ['SISR_RGB_',setTestCur,'_x',num2str(scaleFactor),'_',kernelType_image,'_',kernelType_model]);
    if ~exist(folderResultCur,'file')
        mkdir(folderResultCur);
    end
    
    for i = 1 : length(filepaths)
        
        HR  = imread(fullfile(folderTestCur,filepaths(i).name));
        [~,imageName,ext] = fileparts(filepaths(i).name);
        HR  = modcrop(HR, scaleFactor);
        if size(HR,3)==1
            HR =  cat(3,HR,HR,HR);
        end
        %%% label_RGB (uint8)
        label_RGB = HR;
        %%% LR (uint8)
        LR = imresize_down(HR,scaleFactor,kernelType_image,kernelsigma);
        
        
        HR_ycc = single(rgb2ycbcr(im2double(HR)));
        label  = HR_ycc(:,:,1);
        
        LRY     = im2single(LR);
        
        HR_bic = imresize(LRY,scaleFactor,'bicubic');
        %%% input (single)
        input  = im2single(HR_bic);
        %%% input_RGB (uint8)
        input_RGB = im2uint8(HR_bic);
        
        if useGPU
            input = gpuArray(input);
            LRY   = gpuArray(LRY);
        end
        output = input;
        tic;
        for itern = 1:totalIter
            %%% step 1
            for k = 1:inIter
                output = output + alpha*imresize((LRY - imresize_down(output,scaleFactor,kernelType_model,kernelsigma)),scaleFactor,'bicubic');
            end
            if ns(itern+1)~=ns(itern)
                [net] = loadmodel(modelSigmaS(itern),CNNdenoiser);
                net = vl_simplenn_tidy(net);
                if useGPU
                    net = vl_simplenn_move(net, 'gpu');
                end
            end
            %%% step 2
            res = vl_simplenn(net, output,[],[],'conserveMemory',true,'mode','test');
            im  = res(end).x;
            output = output - im;
        end
        
        if useGPU
            output = gather(output);
        end
        toc;
        output_RGB = im2uint8(output);
        HR_ycc = single(rgb2ycbcr(im2double(output_RGB)));
        output  = HR_ycc(:,:,1);
        
        [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label*255,output*255,ceil(scaleFactor),ceil(scaleFactor)); %%% single
        [PSNRC_Cur,SSIM_Cur_RGB] = Cal_PSNRSSIM(label_RGB,output_RGB,ceil(scaleFactor),ceil(scaleFactor)); %%% single
        
        disp(['Single Image Super-Resolution     ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
        eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),'(',num2str(i),') = PSNR_Cur;']);
        eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),'(',num2str(i),') = PSNRC_Cur;']);
        if showResult
            imshow(cat(1,cat(2,input_RGB,output_RGB),cat(2,(output_RGB-input_RGB),label_RGB)));
            drawnow;
            title(['Single Image Super-Resolution     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
            pause(pauseTime)
            %pause()
            imwrite(output_RGB,fullfile(folderResultCur,[imageName,'_x',num2str(scaleFactor),'.png']));
        end
    end
    disp(['Average PSNR is ',num2str(mean(eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor)])),'%2.2f'),'dB']);
    disp(['Average PSNRC is ',num2str(mean(eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor)])),'%2.4f')]);
    
    %%% save PSNR and SSIM metrics
    save(fullfile(folderResultCur,['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),'.mat']),['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor)])
    save(fullfile(folderResultCur,['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),'.mat']),['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor)])
    
end



