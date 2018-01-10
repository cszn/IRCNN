function y = upsample2(x,K)
for i = 1:size(x,3)
    y(:,:,i) = upsample(upsample(x(:,:,i),K)',K)';
end
end