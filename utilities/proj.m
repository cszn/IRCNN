function out = proj(x,bound)
% x     = input vector
% bound = 2x1 vector
%
% Example: out = proj_bound(x, [1,3]);
% projects a vector x onto the interval [1,3]
% by setting x(x>3) = 3, and x(x<1) = 1
%
% 2016-07-24 Stanley Chan

if ~exist('bound', 'var')
    bound = [0,1];
end
out = min(max(x,bound(1)),bound(2));