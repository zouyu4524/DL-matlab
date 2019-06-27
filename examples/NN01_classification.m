%% Create a simple classification neural network using
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

% create serveral dense (fullyConnectedLayer) layers
fc1 = fullyConnectedLayer(64);
% fc1.Weights = randn([64, 2])*0.1;
fc2 = fullyConnectedLayer(64);
% fc2.Weights = randn([64, 64])*0.1;
fc3 = fullyConnectedLayer(2);
% fc3.Weights = randn([2, 64])*0.1;

% construct the network by stacking layers
layers = [
    imageInputLayer([1, 1, 2])
    fc1
    reluLayer()
    fc2
    reluLayer()
    fc3
    softmaxLayer()
    classificationLayer()
];

% generate dataset
dataset_classification_generator;
load('../dataset/classification.mat');

% set training options
opts = trainingOptions('sgdm', ...
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
saveTrainingProcess('NN01_classification_training_process')