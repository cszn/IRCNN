function y = downsample2(x,K)

for i = 1:size(x,3)
y(:,:,i) = downsample(downsample(x(:,:,i),K)',K)';
end

end