%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of IRCNN for image super-resolution where the latent HR image x is blurred and then downsampled to get the LR image y
% (y can be corrupted by additive Gaussian noise of level Isigma).
%
% The details of this degradation can be found by the following paper.
% [1] S. H. Chan, X. Wang, and O. A. Elgendy "Plug-and-Play ADMM for image restoration: Fixed point convergence and applications", IEEE Transactions on Computational Imaging, 2016.
%
% The objective function is given by min_x 1/(Isigma^2)||x*k_{direct downsampler with scale factor sf}-y||^2 + lamda Phi(x)
%
%                  k  --  blur kernel, not limited to Gaussian blur
% direct downsampler  --  implemented by matlab function "downsample",
%                 sf  --  scale factor, 2,3,4,...
%             Isigma  --  estimated noise level of y, should be slightly larger than the true one.
%
% @inproceedings{zhang2017learning,
%   title={Learning Deep CNN Denoiser Prior for Image Restoration},
%   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%   year={2017}
% }

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear; clc;

addpath('utilities');

imageSets   = {'Set5','Set14','BSD100','Urban100'}; % testing dataset

%%% setting
setTest     = imageSets([1]); % select the dataset
showResult  = 1;
pauseTime   = 0;
useGPU      = 1; % 1 or 0, true or false

folderTest  = 'testsets';
folderResult= 'results';
taskTestCur = 'SISR';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%% parameter setting of HQS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Important!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sf          = 3; % scale factor
Isigma      = 0.5/255; % noise level of y (Note: for RGB images, y is the Y channel of YCbCr space.) from [0.5, 50]/255
Isigma      = max(Isigma,0.1/255);
Msigma      = sf;    % noise level of last denoiser, from [1,15]
% blur kernel k, not limited to Gaussian blur
kernelsigma = 1.6; % width (sigma) of the Gaussian blur kernel
% from [0.6 2.4], e.g., sf = 2, kernelsigma = 1; sf = 3, kernelsigma = 1.6; sf = 4, kernelsigma = 2;
k       = fspecial('gaussian',7,kernelsigma);

%k       = fspecial('motion',20,45); % You can try this motion blur kernel ^_^
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% default parameter setting of HQS
totalIter   = 30;          % default 30
modelSigmaS = logspace(log10(49),log10(Msigma),totalIter);   % default 49, or 3*sf
ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];

lamda       = (Isigma^2)/3; % default 3, ****** from {1 2 3 4} ******


%% load denoiser model
folderModel = 'models';
load(fullfile(folderModel,'modelgray.mat'));


%% do SISR
for n_set = 1 : numel(setTest)
    % read images
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
    eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(sf),' = zeros(length(filepaths),1);']);
    eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(sf),' = zeros(length(filepaths),1);']);
    
    % folder to store results
    folderResultCur = fullfile(folderResult, ['SISR_YCbCr_direct_downsample_',setTestCur,'_x',num2str(sf)]);
    if ~exist(folderResultCur,'file')
        mkdir(folderResultCur);
    end
    
    for i = 1 : length(filepaths)
        
        HR  = imread(fullfile(folderTestCur,filepaths(i).name));
        [~,imageName,ext] = fileparts(filepaths(i).name);
        HR  = modcrop(HR, sf);
        
        % label_RGB (uint8)
        label_RGB = HR;
        chanel = size(HR,3);
        
        %%%%%%%%%%%%%%%%%%%%%% image degradation %%%%%%%%%%%%%%%%%%%%
        % LR (uint8), get the LR image
        blur_HR = imfilter(HR,k,'circular'); % blurred
        LR      = downsample2(blur_HR, sf);  % downsampled
        LR      = uint8(LR);
        
        
        if chanel == 3
            % label (single)
            HR_ycc = single(rgb2ycbcr(im2double(HR)));
            label  = HR_ycc(:,:,1);
            LR_ycc = single(rgb2ycbcr(im2double(LR)));
            LRY    = LR_ycc(:,:,1);
            % input (single)
            HR_bic     = imresize(im2double(LR),sf,'bicubic');
            LR_bic_ycc = rgb2ycbcr(HR_bic);
            input      = im2single(LR_bic_ycc(:,:,1));
            % input_RGB (uint8)
            input_RGB  = im2uint8(HR_bic);
        else
            % label (single)
            label  = im2single(HR);
            LRY    = im2single(LR);
            HR_bic = imresize(LRY,sf,'bicubic');
            % input (single)
            input  = im2single(HR_bic);
            % input_RGB (uint8)
            input_RGB = HR_bic;
        end
        
        y = im2single(LRY);
        [rows_in,cols_in] = size(y);
        rows      = rows_in*sf;
        cols      = cols_in*sf;
        [G,Gt]    = defGGt(k,sf);
        GGt       = constructGGt(k,sf,rows,cols);
        Gty       = Gt(y);
        
        if useGPU
            input = gpuArray(input);
            LRY   = gpuArray(LRY);
        end
        output = input;
        tic;
        for itern = 1:totalIter
            
            % step 1, closed-form solution, see Chan et al. [1] for details
            rho    = lamda*255^2/(modelSigmaS(itern)^2);
            rhs    = Gty + rho*output;
            output = (rhs - Gt(real(ifft2(fft2(G(rhs))./(GGt + rho)))))/rho;
            
            % load denoiser
            if ns(itern+1)~=ns(itern)
                [net] = loadmodel(modelSigmaS(itern),CNNdenoiser);
                net = vl_simplenn_tidy(net);
                if useGPU
                    net = vl_simplenn_move(net, 'gpu');
                end
            end
            
            % step 2, perform denoising
            res = vl_simplenn(net, output,[],[],'conserveMemory',true,'mode','test');
            im = res(end).x; % residual image
            output = output - im;
            % imshow(output)
            % drawnow;
            % pause(1)
        end
        
        
        if useGPU
            output = gather(output);
        end
        toc;
        if chanel == 3
            % output_RGB (uint8)
            LR_bic_ycc(:,:,1) = double(output);
            output_RGB = im2uint8(ycbcr2rgb(LR_bic_ycc));
        else
            % output_RGB (uint8)
            output_RGB = im2uint8(output);
        end
        
        [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label*255,output*255,ceil(sf),ceil(sf)); % calculate PSNR and SSIM on Y channel of YCbCr space
        [PSNRC_Cur,SSIM_Cur_RGB] = Cal_PSNRSSIM(label_RGB,output_RGB,ceil(sf),ceil(sf)); % calculate PSNR and SSIM on R,G,B channels
        
        disp(['Single Image Super-Resolution     ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
        eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = PSNR_Cur;']);
        eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = PSNRC_Cur;']);
        if showResult
            imshow(cat(2,input_RGB,output_RGB,label_RGB));
            drawnow;
            title(['Single Image Super-Resolution     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
            pause(pauseTime)
            %pause()
            imwrite(output_RGB,fullfile(folderResultCur,[imageName,'_x',num2str(sf),'.png']));
        end
    end
    disp(['Average PSNR on  Y  is ',num2str(mean(eval(['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(sf)])),'%2.2f'),'dB']);
    disp(['Average PSNR on RGB is ',num2str(mean(eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(sf)])),'%2.2f'),'dB']);
    
    % save PSNR and SSIM metrics
    save(fullfile(folderResultCur,['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(sf),'.mat']),['PSNR_',taskTestCur,'_',setTestCur,'_x',num2str(sf)])
    save(fullfile(folderResultCur,['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(sf),'.mat']),['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(sf)])
    
end



