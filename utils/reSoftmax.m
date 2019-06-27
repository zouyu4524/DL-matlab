function Y = reSoftmax(X, dim)
% apply softmax to X along specified dimension
% X: input data
% dim: dimension to apply softmax
X = X - max(X,[],dim);
X = exp(X);
Y = X./sum(X,dim);