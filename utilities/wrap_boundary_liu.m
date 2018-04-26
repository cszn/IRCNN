function ret = wrap_boundary_liu(img, img_size)
% wrap_boundary_liu.m
%
%   pad image boundaries such that image boundaries are circularly smooth
%
%     written by Sunghyun Cho (sodomau@postech.ac.kr)
%
% This is a variant of the method below:
%   Reducing boundary artifacts in image deconvolution
%     Renting Liu, Jiaya Jia
%     ICIP 2008
%
    [H, W, Ch] = size(img);
    H_w = img_size(1) - H;
    W_w = img_size(2) - W;

    ret = zeros(img_size(1), img_size(2), Ch);
    for ch = 1:Ch
        alpha = 1;
        HG = img(:,:,ch);

        r_A = zeros(alpha*2+H_w, W);
        r_A(1:alpha, :) = HG(end-alpha+1:end, :);
        r_A(end-alpha+1:end, :) = HG(1:alpha, :);
        a = ((1:H_w)-1)/(H_w-1);
        r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1);
        r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end);

        A2 = solve_min_laplacian(r_A(alpha:end-alpha+1,:));
        r_A(alpha:end-alpha+1,:) = A2;
        A = r_A;

        r_B = zeros(H, alpha*2+W_w);
        r_B(:, 1:alpha) = HG(:, end-alpha+1:end);
        r_B(:, end-alpha+1:end) = HG(:, 1:alpha);
        a = ((1:W_w)-1)/(W_w-1);
        r_B(1, alpha+1:end-alpha) = (1-a)*r_B(1,alpha) + a*r_B(1,end-alpha+1);
        r_B(end, alpha+1:end-alpha) = (1-a)*r_B(end,alpha) + a*r_B(end,end-alpha+1);

        B2 = solve_min_laplacian(r_B(:, alpha:end-alpha+1));
        r_B(:,alpha:end-alpha+1,:) = B2;
        B = r_B;

        r_C = zeros(alpha*2+H_w, alpha*2+W_w);
        r_C(1:alpha, :) = B(end-alpha+1:end, :);
        r_C(end-alpha+1:end, :) = B(1:alpha, :);
        r_C(:, 1:alpha) = A(:, end-alpha+1:end);
        r_C(:, end-alpha+1:end) = A(:, 1:alpha);

        C2 = solve_min_laplacian(r_C(alpha:end-alpha+1, alpha:end-alpha+1));
        r_C(alpha:end-alpha+1, alpha:end-alpha+1) = C2;
        C = r_C;

        A = A(alpha:end-alpha-1, :);
        B = B(:, alpha+1:end-alpha);
        C = C(alpha+1:end-alpha, alpha+1:end-alpha);

        ret(:,:,ch) = [img(:,:,ch) B; A C];
    end

end


function [img_direct] = solve_min_laplacian(boundary_image)
% function [img_direct] = poisson_solver_function(gx,gy,boundary_image)
% Inputs; Gx and Gy -> Gradients
% Boundary Image -> Boundary image intensities
% Gx Gy and boundary image should be of same size
    [H,W] = size(boundary_image);

    % Laplacian
    f = zeros(H,W);                          clear j k

    % boundary image contains image intensities at boundaries
    boundary_image(2:end-1, 2:end-1) = 0;
    j = 2:H-1;      k = 2:W-1;      f_bp = zeros(H,W);
    f_bp(j,k) = -4*boundary_image(j,k) + boundary_image(j,k+1) + ...
        boundary_image(j,k-1) + boundary_image(j-1,k) + boundary_image(j+1,k);
    clear j k

    %f1 = f - reshape(f_bp,H,W); % subtract boundary points contribution
    f1 = f - f_bp; % subtract boundary points contribution
    clear f_bp f

    % DST Sine Transform algo starts here
    f2 = f1(2:end-1,2:end-1);                   clear f1
    % compute sine tranform
    tt = dst(f2);       f2sin = dst(tt')';      clear f2

    % compute Eigen Values
    [x,y] = meshgrid(1:W-2, 1:H-2);
    denom = (2*cos(pi*x/(W-1))-2) + (2*cos(pi*y/(H-1)) - 2);

    % divide
    f3 = f2sin./denom;                          clear f2sin x y

    % compute Inverse Sine Transform
    tt = idst(f3);      clear f3;       img_tt = idst(tt')';        clear tt

    % put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image;
    img_direct(2:end-1,2:end-1) = 0;
    img_direct(2:end-1,2:end-1) = img_tt;
end
