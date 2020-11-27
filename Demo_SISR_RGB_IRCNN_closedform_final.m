
addpath('utilities');
imageSets   = {'set68','Set14'}; % testing dataset

setTest     = imageSets([1]); % select the dataset
showResult  = 1;
pauseTime   = 0;
useGPU      = 0; % 1 or 0, true or false

folderTest  = 'testsets';
folderResult= 'results';
taskTestCur = 'SISR';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end


for scaleFactor = 2:4
    totalIter   = 8;
    inIter      = 5;
    alpha       = 1.75;
    Isigma      = 0.5/255; % default 0.5/255 for noise-free case. It should be larger than noisesigma, e.g., Isigma = noisesigma + 2/255;
    Isigma      = max(Isigma,0.1/255);
    modelSigmaS = logspace(log10(12*scaleFactor),log10(scaleFactor),totalIter);
    ns          = min(25,max(ceil(modelSigmaS/2),1));
    ns          = [ns(1)-1,ns];
    
    lamda       = (Isigma^2)/3; % default 3, ****** from {1 2 3 4} ******
    
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
        for ii = 1 : length(ext)
            filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{ii})));
        end
        
        eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),' = zeros(length(filepaths),10);']);
        
        %%% folder to store results
        folderResultCur = fullfile(folderResult, ['SISR_RGB_',setTestCur,'_x',num2str(scaleFactor)]);
        if ~exist(folderResultCur,'file')
            mkdir(folderResultCur);
        end
        
        load kernels.mat
        for j = 1:12
            kernel = double(kernels{j});
            
            for i = 1 : length(filepaths)
                
                
                d = scaleFactor;
                dr = d;
                dc = d;
                Nb = dr*dc;
                
                
                HR  = imread(fullfile(folderTestCur,filepaths(i).name));
                [~,imageName,ext] = fileparts(filepaths(i).name);
                HR  = modcrop(HR, scaleFactor);
                if size(HR,3)==1
                    HR = cat(3,HR,HR,HR);
                end
                %%% label_RGB (uint8)
                label_RGB = HR;

                blur_HR   = imfilter(HR,kernel,'circular','conv'); % blurred
                LR        = downsample2(blur_HR, d);
                LRY     = im2single(LR);
                
                m = size(HR,1);
                n = size(HR,2);

                B = kernel;
                FB = psf2otf(B,[m,n]);
                FBC = conj(FB);
                F2B = abs(FB).^2;
                
                [nr,nc,ccc] = size(LRY);
                m = nr*nc;
                HR_bic = imresize(LRY,scaleFactor,'nearest');
                
                STy = zeros(size(HR_bic));
                STy(1:d:end,1:d:end,:)=LRY;
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
                    rho    = lamda*255^2/(modelSigmaS(itern)^2);
                    tau = rho;

                    FR = FBC.*fft2(STy) + fft2(tau*output);
                    
                    for p = 1:3
                        output(:,:,p) = INVLS(FB,FBC,F2B,FR(:,:,p),tau,Nb,nr,nc,m);
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
                
                [PSNRC_Cur,SSIM_Cur_RGB] = Cal_PSNRSSIM(label_RGB,output_RGB,ceil(scaleFactor*scaleFactor),ceil(scaleFactor*scaleFactor)); %%% single
                
                psnrs(scaleFactor-1,j,i) = PSNRC_Cur;
                
                
                disp(['Single Image Super-Resolution     ',num2str(PSNRC_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
                eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor),'(',num2str(i),',',num2str(j),')= PSNRC_Cur;']);
                if showResult
                    imshow(cat(1,cat(2,input_RGB,output_RGB),cat(2,(output_RGB-input_RGB),label_RGB)));
                    drawnow;
                    title(['Single Image Super-Resolution     ',filepaths(i).name,'    ',num2str(PSNRC_Cur,'%2.2f'),'dB'],'FontSize',12)
                    pause(pauseTime)
                    %pause()
                    imwrite(output_RGB,fullfile(folderResultCur,[imageName,'_x',num2str(scaleFactor),'_k_',num2str(j),'.png']));
                end
            end
            
            disp(['Average PSNRC is ',num2str(mean(eval(['PSNRC_',taskTestCur,'_',setTestCur,'_x',num2str(scaleFactor)])),'%2.4f')]);
        end
        %%% save PSNR and SSIM metrics
        
        save(fullfile(folderResultCur,'psnrs.mat'),psnrs)
        
    end
    
end

