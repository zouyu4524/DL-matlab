%% dataset: another regression generation
% this dataset is to test multiple input layer

% set random seed for reproduction
seed = 1;
rng(seed);

% check fold existence, make one if not exist
if ~exist('../dataset', 'dir')
    mkdir('../dataset')
end

% generate X randomly, 4-D double, the last dimension is the number of
% entries, we use 1x1x[2+3] type to fit the imageInputLayer of the deep
% learning toolbox, which accepts inputs in HxWxC type, where H, W and C
% represents height, width and channel of the image, respectively. Here we
% put the feature dimension of the input data at the channel dimension.

% to mimic two inputs, we separate X into two branches, one has two
% features and another has three features. We multiply 2 to the second
% branch
X = rand(1, 1, 5, 10000);
X(1,1,3:5,:) = 2*X(1,1,3:5,:);

X_reshape = reshape(X, 5, 10000);

% apply functions to produce Y
Y = [mean(X_reshape); std(X_reshape); var(X_reshape)]';
Y = reSoftmax(Y, 2);

% save the dataset
save('../dataset/regression04.mat', 'X', 'Y', 'seed');