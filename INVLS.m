function [Xest,FX] = INVLS(FB,FBC,F2B,FR,tau,Nb,nr,nc,m,varargin)
%**************************************************************************
% Author: Ningning ZHAO (2015 Oct.)
% University of Toulouse, IRIT/INP-ENSEEIHT
% Email: buaazhaonn@gmail.com
%
% USAGE: Analytical solution as below
%        x = (B^H S^H SH + tau I )^(-1) R
% INPUT:
    %  FB-> Fourier transform of the blurring kernel B
    %  FBC->conj(FB)
    %  F2B->abs(FB)^2
    %  FR-> Fourier transform of R
    %  Nb-> scale factor Nb = dr*dc
    %  nr,nc-> size of the observation
    %  m-> No. of the pixels of the observation m = nr*nc    
% OUTPUT:
    %  Xest->Analytical solution
    %  FX->Fourier transform of the analytical solution
%************************************************************************** 
% if nargin ==9
%     F2D = 1;  
% elseif nargin==10
%     F2D = varargin{1}; % TV prior: F2D = F2DH + F2DV +c
% end

x1 = FB.*FR;

FBR = BlockMM(nr,nc,Nb,m,x1);
invW = BlockMM(nr,nc,Nb,m,F2B);
invWBR = FBR./(invW + tau);

fun = @(block_struct) block_struct.data.*invWBR;
FCBinvWBR = blockproc(FBC,[nr,nc],fun);
FX = (FR-FCBinvWBR)/tau;

Xest = real(ifft2(FX));

% clip 4 patches, and then sum
function x = BlockMM(nr,nc,Nb,m,x1)
myfun = @(block_struct) reshape(block_struct.data,m,1);
x1 = blockproc(x1,[nr nc],myfun);

x1 = reshape(x1,m,Nb);
x1 = mean(x1,2);
x = reshape(x1,nr,nc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
function [Xest,FX] = INVLS2(FB,FBC,F2B,FR,tau,Nb,nr,nc,m,varargin)
%**************************************************************************
% Author: Ningning ZHAO (2015 Oct.)
% University of Toulouse, IRIT/INP-ENSEEIHT
% Email: buaazhaonn@gmail.com
%
% USAGE: Analytical solution as below
%        x = (B^H S^H SH + tau I )^(-1) R
% INPUT:
    %  FB-> Fourier transform of the blurring kernel B
    %  FBC->conj(FB)
    %  F2B->abs(FB)^2
    %  FR-> Fourier transform of R
    %  Nb-> scale factor Nb = dr*dc
    %  nr,nc-> size of the observation
    %  m-> No. of the pixels of the observation m = nr*nc    
% OUTPUT:
    %  Xest->Analytical solution
    %  FX->Fourier transform of the analytical solution
%************************************************************************** 
% if nargin ==9
%     F2D = 1;  
% elseif nargin==10
%     F2D = varargin{1}; % TV prior: F2D = F2DH + F2DV +c
% end

x1 = FB.*FR;

FBR = BlockMM2(nr,nc,Nb,m,x1);
invW = BlockMM2(nr,nc,Nb,m,F2B);
invWBR = FBR./(invW + tau*Nb);

% fun = @(block_struct) block_struct.data.*invWBR;
% FCBinvWBR = blockproc(FBC,[nr,nc],fun);

FBC1 = vl_nnSubP(FBC,[],'scale',1/sqrt(Nb));
FCBinvWBR = vl_nnSubP(FBC1.*invWBR,[],'scale',sqrt(Nb));

FX = (FR-FCBinvWBR)/tau;

Xest = real(ifft2(FX));


% clip 4 patches, and then sum
function x = BlockMM2(nr,nc,Nb,m,x1)
x = vl_nnSubP(x1,[],'scale',1/sqrt(Nb));
x = sum(x,3);

