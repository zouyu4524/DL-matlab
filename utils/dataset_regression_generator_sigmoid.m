%% dataset: another regression generation
% this dataset is to test sigmoid activation function

% set random seed for reproduction
seed = 1;
rng(seed);

% check fold existence, make one if not exist
if ~exist('../dataset', 'dir')
    mkdir('../dataset')
end

% generate X randomly, 4-D double, the last dimension is the number of
% entries, we use 1x1x2 type to fit the imageInputLayer of the deep
% learning toolbox, which accepts inputs in HxWxC type, where H, W and C
% represents height, width and channel of the image, respectively. Here we
% put the feature dimension of the input data at the channel dimension.
X = rand(1, 1, 2, 10000);

X_reshape = reshape(X, 2, 10000)';
x1 = X_reshape(:, 1); x2 = X_reshape(:, 2);

% apply functions to produce Y
% apply functions to produce Y
y1 = @(x1, x2) (x1 + x2)/2;
y2 = @(x1, x2) x1.*x2;
y3 = @(x1, x2) (x1.^2 + x2.^2)/2;
Y = [y1(x1, x2), y2(x1, x2), y3(x1, x2)];

% save the dataset
save('../dataset/regression02.mat', 'X', 'Y', 'seed');