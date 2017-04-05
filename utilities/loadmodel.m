


function [net,n] = loadmodel(modelsigma,models)
n = min(25,max(ceil(modelsigma/2),1));
net.layers = [models{n}];