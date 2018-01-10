function GGt = constructGGt(h,K,rows,cols)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Eigen-decomposition for super-resolution
% Stanley Chan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hth = conv2(h,rot90(h,2));

yc = ceil(size(hth,1)/2);  % mark the center coordinate
xc = ceil(size(hth,2)/2);

L = floor(size(hth,1)/K);  % width of the new filter 
                           %  = (1/k) with of the original filter
                           
g = zeros(L,L);            % initialize new filter
for i=-floor(L/2):floor(L/2)
    for j=-floor(L/2):floor(L/2)
        g(i+floor(L/2)+1,j+floor(L/2)+1) = hth(yc+K*i, xc+K*j);
    end
end

GGt = abs(fft2(g,rows/K,cols/K));