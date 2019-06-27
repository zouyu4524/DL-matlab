%% Create a simple regression neural network using
% deep learning toolbox
% NOTE: the compatibality problem of initializer of the layers
%       before R2019a, the fullyConnectedLayer is initilized by normal
%       distribution with zero mean and 0.01 variance by default regardless
%       of the input and output size of the layer. This is changed since
%       R2019a, which is replaced by 'xavier' initializer [1]
%       [1]. https://ww2.mathworks.cn/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html
% The 'xavier' initializer helps stablize training.

% add path
addpath('../utils')

hidden_units = 64; % number of hidden layer units

% create layers
input = imageInputLayer([1, 1, 2], 'Name', 'input');
fc1 = fullyConnectedLayer(hidden_units, 'Name', 'hidden1');
fc2 = fullyConnectedLayer(hidden_units, 'Name', 'hidden2');
fc3 = fullyConnectedLayer(3, 'Name', 'ready2output');
output = regressionLayer('Name', 'output');

layers = [
    input
    fc1
    reluLayer('Name', 'h1_activation')
    fc2
    reluLayer('Name', 'h2_activation')
    fc3
    tanhLayer('Name', 'output_activation')
    output
];

% generate dataset
dataset_regression_generator;
load('../dataset/regression01.mat');

% set training options
opts = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'Verbose', true, ...
    'Plots','training-progress');

% train the network
net = trainNetwork(X, Y, layers, opts);

% save training process
saveTrainingProcess('NN02_regression_training_process')