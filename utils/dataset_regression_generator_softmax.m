%% dataset: another regression generation
% this dataset is to test softmax activation function

% set random seed for reproduction
seed = 1;
rng(seed);

% check fold existence, make one if not exist
if ~exist('../dataset', 'dir')
    mkdir('../dataset')
end

% generate X randomly, 4-D double, the last dimension is the number of
% entries, we use 1x1x10 type to fit the imageInputLayer of the deep
% learning toolbox, which accepts inputs in HxWxC type, where H, W and C
% represents height, width and channel of the image, respectively. Here we
% put the feature dimension of the input data at the channel dimension.
X = rand(1, 1, 10, 10000);

X_reshape = reshape(X, 10, 10000);

% apply functions to produce Y
Y = [mean(X_reshape); std(X_reshape); var(X_reshape)]';
Y = reSoftmax(Y, 2);

% save the dataset
save('../dataset/regression03.mat', 'X', 'Y', 'seed');