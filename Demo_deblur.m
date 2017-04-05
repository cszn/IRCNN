% (non-blind) image deblurring

% @inproceedings{zhang2017learning,
%   title={Learning Deep CNN Denoiser Prior for Image Restoration},
%   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%   year={2017}
% }

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

clear; clc;

addpath('utilities');
imageSets    = {'Set3G','Set3C'}; %%% testing dataset
setTest      = imageSets([1]); %%% select the dataset

showResult   = 1;
pauseTime    = 1;
useGPU       = 1;

folderTest   = 'testsets';
folderResult = 'results';
folderModel  = 'models';
taskTestCur  = 'Deblur';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

load(fullfile('kernels','Levin09.mat'));
kernelType = 1; % 1~8
if kernelType > 8
    k = fspecial('gaussian', 25, 1.6);
else
    k = kernels{kernelType};
end
sigmas      = [2, 2.55, 7.65]/255;
sigma       = sigmas(3);
totalIter   = 30; % default
lamda       = (sigma^2)/3; % default 3, ****** from {1 2 3 4} ******
modelSigma1 = 49; % default
modelSigma2 = 13; % ****** from {1 3 5 7 9 11 13 15} ******
modelSigmaS = logspace(log10(modelSigma1),log10(modelSigma2),totalIter);
rho         = sigma^2/((modelSigma1/255)^2);

ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];

for n_set = 1 : numel(setTest)
    %%% read images
    setTestCur = cell2mat(setTest(n_set));
    disp('--------------------------------------------');
    disp(['----',setTestCur,'-----Image Debluring-----']);
    disp('--------------------------------------------');
    folderTestCur = fullfile(folderTest,setTestCur);
    ext                 =  {'*.jpg','*.png','*.bmp'};
    filepaths           =  [];
    for i = 1 : length(ext)
        filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
    end
    eval(['PSNR_',taskTestCur,'_',setTestCur,' = zeros(length(filepaths),1);']);
    
    %%% folder to store results
    folderResultCur = fullfile(folderResult, ['Deblur_',setTestCur,'_kernel_',num2str(kernelType)]);
    if ~exist(folderResultCur,'file')
        mkdir(folderResultCur);
    end
    
    for i = 1 : length(filepaths)
        
        
        x  = imread(fullfile(folderTestCur,filepaths(i).name));
        [~,imageName,ext] = fileparts(filepaths(i).name);
        randn('seed',0);
        y = imfilter(im2double(x), k, 'circular', 'conv') + sigma*randn(size(x));
        [w,h,c]  = size(y);
        V = psf2otf(k,[w,h]);
        denominator = abs(V).^2;
        
        if c>1
            denominator = repmat(denominator,[1,1,c]);
            V = repmat(V,[1,1,c]);
        end
        upperleft   = conj(V).*fft2(y);
        
        if c==1
            load(fullfile(folderModel,'modelgray.mat'));
        elseif c==3
            load(fullfile(folderModel,'modelcolor.mat'));
        end
        z = single(y);
        if useGPU
            z           = gpuArray(z);
            upperleft   = gpuArray(upperleft);
            denominator = gpuArray(denominator);
        end
        tic;
        for itern = 1:totalIter
            %%% step 1
            rho = lamda*255^2/(modelSigmaS(itern)^2);
            z = real(ifft2((upperleft + rho*fft2(z))./(denominator + rho)));
            if ns(itern+1)~=ns(itern)
                [net] = loadmodel(modelSigmaS(itern),CNNdenoiser);
                net = vl_simplenn_tidy(net);
                if useGPU
                    net = vl_simplenn_move(net, 'gpu');
                end
            end
            %%% step 2
            res = vl_simplenn(net, z,[],[],'conserveMemory',true,'mode','test');
            residual = res(end).x;
            z = z - residual;
        end
        
        if useGPU
            output = im2uint8(gather(z));
        end
        toc;
        
        [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(x,output,0,0); %%% single
        disp(['Image Deblurring     ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
        eval(['PSNR_',taskTestCur,'_',setTestCur,'(',num2str(i),') = PSNR_Cur;']);
        
        if showResult
            imshow(cat(2,im2uint8(y),output,x));
            drawnow;
            title(['Image Deblurring     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
            pause(pauseTime)
            %pause()
            imwrite(output,fullfile(folderResultCur,[imageName,'.png']));
        end
    end
    disp(['Average PSNR is ',num2str(mean(eval(['PSNR_',taskTestCur,'_',setTestCur])),'%2.2f'),'dB']);
    
    %%% save PSNR
    save(fullfile(folderResultCur,['PSNR_',taskTestCur,'_',setTestCur,'.mat']),['PSNR_',taskTestCur,'_',setTestCur])
    
    
end














