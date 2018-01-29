function out = linearlcc(in, quantization)
%  @inproceedings{zhang2017learning,
%    title={Learning Deep CNN Denoiser Prior for Image Restoration},
%    author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
%    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%    pages={3929--3938},
%    year={2017},
%  }
%quantization = 0;
% in : uint8; RGB or Bayer pattern [0~255];
% out: double/uint8: RGB [0~255];

[m,n,ch]=size(in);
% if ch==3
%     B=zeros(m,n);
%     B(1:2:m,1:2:n)=in(1:2:m,1:2:n,2);
%     B(2:2:m,2:2:n)=in(2:2:m,2:2:n,2);
%     B(1:2:m,2:2:n)=in(1:2:m,2:2:n,1);
%     B(2:2:m,1:2:n)=in(2:2:m,1:2:n,3);
%     in=B;
%     clear B;
% end

in = double(in);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function out = lcc1(in)
% returns the color interpolated image using Laplacian
% second-order color correction I
%
% Assumptions : in has following color patterns
%
%  ------------------> grbg
%  |  G R G R ...
%  |  B G B G ...
%  |  G R G R ...
%  |  B G B G ...
%  |  . . . . .

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inR = in; inG = in; inB = in;
out = in;
outR = inR; outG = inG; outB = inB;

% G channel
for i=4:2:m-2,
    for j=3:2:n-3,
        delta_H = abs(inB(i,j-2)+inB(i,j+2)-2*inB(i,j))+abs(inG(i,j-1)-inG(i,j+1));
        delta_V = abs(inB(i-2,j)+inB(i+2,j)-2*inB(i,j))+abs(inG(i-1,j)-inG(i+1,j));
        if delta_H < delta_V,
            outG(i,j) = 1/2*(inG(i,j-1)+inG(i,j+1))+1/4*(2*inB(i,j)-inB(i,j-2)-inB(i,j+2));
        elseif delta_H > delta_V,
            outG(i,j) = 1/2*(inG(i-1,j)+inG(i+1,j))+1/4*(2*inB(i,j)-inB(i-2,j)-inB(i+2,j));
        else
            outG(i,j) = 1/4*(inG(i,j-1)+inG(i,j+1)+inG(i-1,j)+inG(i+1,j))+1/8*(4*inB(i,j)-inB(i,j-2)-inB(i,j+2)-inB(i-2,j)-inB(i+2,j));
        end
    end
end

for i=3:2:m-3,
    for j=4:2:n-2,
        delta_H = abs(inR(i,j-2)+inR(i,j+2)-2*inR(i,j))+abs(inG(i,j-1)-inG(i,j+1));
        delta_V = abs(inR(i-2,j)+inR(i+2,j)-2*inR(i,j))+abs(inG(i-1,j)-inG(i+1,j));
        if delta_H < delta_V,
            outG(i,j) = 1/2*(inG(i,j-1)+inG(i,j+1))+1/4*(2*inR(i,j)-inR(i,j-2)-inR(i,j+2));
        elseif delta_H > delta_V,
            outG(i,j) = 1/2*(inG(i-1,j)+inG(i+1,j))+1/4*(2*inR(i,j)-inR(i-2,j)-inR(i+2,j));
        else
            outG(i,j) = 1/4*(inG(i,j-1)+inG(i,j+1)+inG(i-1,j)+inG(i+1,j))+1/8*(4*inR(i,j)-inR(i,j-2)-inR(i,j+2)-inR(i-2,j)-inR(i+2,j));
        end
    end
end

% R channel
for i=1:2:m-1,
    outR(i,3:2:n-1) = 1/2*(inR(i,2:2:n-2)+inR(i,4:2:n))+1/4*(2*outG(i,3:2:n-1)-outG(i,2:2:n-2)-outG(i,4:2:n));
end

for i=2:2:m-2,
    outR(i,2:2:n) = 1/2*(inR(i-1,2:2:n)+inR(i+1,2:2:n))+1/4*(2*outG(i,2:2:n)-outG(i-1,2:2:n)-outG(i+1,2:2:n));
end

for i=2:2:m-2,
    for j=3:2:n-1,
        delta_P = abs(inR(i-1,j+1)-inR(i+1,j-1))+abs(2*outG(i,j)-outG(i-1,j+1)-outG(i+1,j-1));
        delta_N = abs(inR(i-1,j-1)-inR(i+1,j+1))+abs(2*outG(i,j)-outG(i-1,j-1)-outG(i+1,j+1));
        if delta_N < delta_P,
            outR(i,j) = 1/2*(inR(i-1,j-1)+inR(i+1,j+1))+1/2*(2*outG(i,j)-outG(i-1,j-1)-outG(i+1,j+1));
        elseif delta_N > delta_P,
            outR(i,j) = 1/2*(inR(i-1,j+1)+inR(i+1,j-1))+1/2*(2*outG(i,j)-outG(i-1,j+1)-outG(i+1,j-1));
        else
            outR(i,j) = 1/4*(inR(i-1,j-1)+inR(i-1,j+1)+inR(i+1,j-1)+inR(i+1,j+1))+1/4*(4*outG(i,j)-outG(i-1,j-1)-outG(i-1,j+1)-outG(i+1,j-1)-outG(i+1,j+1));
        end
    end
end

% B channel
for i=2:2:m,
    outB(i,2:2:n-2) = 1/2*(inB(i,1:2:n-3)+inB(i,3:2:n-1))+1/4*(2*outG(i,2:2:n-2)-outG(i,1:2:n-3)-outG(i,3:2:n-1));
end

for i=3:2:m-1,
    outB(i,1:2:n-1) =  1/2*(inB(i-1,1:2:n-1)+inB(i+1,1:2:n-1))+1/4*(2*outG(i,1:2:n-1)-outG(i-1,1:2:n-1)-outG(i+1,1:2:n-1));
end

for i=3:2:m-1,
    for j=2:2:n-2,
        delta_P = abs(inB(i-1,j+1)-inB(i+1,j-1))+abs(2*outG(i,j)-outG(i-1,j+1)-outG(i+1,j-1));
        delta_N = abs(inB(i-1,j-1)-inB(i+1,j+1))+abs(2*outG(i,j)-outG(i-1,j-1)-outG(i+1,j+1));
        if delta_N < delta_P,
            outB(i,j) = 1/2*(inB(i-1,j-1)+inB(i+1,j+1))+1/2*(2*outG(i,j)-outG(i-1,j-1)-outG(i+1,j+1));
        elseif delta_N > delta_P,
            outB(i,j) = 1/2*(inB(i-1,j+1)+inB(i+1,j-1))+1/2*(2*outG(i,j)-outG(i-1,j+1)-outG(i+1,j-1));
        else
            outB(i,j) = 1/4*(inB(i-1,j-1)+inB(i-1,j+1)+inB(i+1,j-1)+inB(i+1,j+1))+1/4*(4*outG(i,j)-outG(i-1,j-1)-outG(i-1,j+1)-outG(i+1,j-1)-outG(i+1,j+1));
        end
    end
end


out(:,:,1) = outR;
out(:,:,2) = outG;
out(:,:,3) = outB;



if quantization
    out = uint8(out);
end







