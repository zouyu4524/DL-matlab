%% Create a simple regression neural network using
% deep learning toolbox, with two inputs
% NOTE: the compatibality problem of initializer of the layers
%       before R2019a, the fullyConnectedLayer is initilized by normal
%       distribution with zero mean and 0.01 variance by default regardless
%       of the input and output size of the layer. This is changed since
%       R2019a, which is replaced by 'xavier' initializer [1]
%       [1]. https://ww2.mathworks.cn/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html
% The 'xavier' initializer helps stablize training.

% The deep learning toolbox doesn't support multiple inputs in either
% SeriesNetwork or DAGNetwork for now (R2019a), we develop an input layer
% to support this demand for a temporarily workaround.

% add path
addpath('../utils')
addpath('../layers')

hidden_units = 64; % number of hidden layer units

% create input layer that accepts two inputs, one has two features and
% another has three features
[lgraph, inputNames] = multiInputsLayer([2, 3], {'input1', 'input2'});

% create layers before concatenation
% one dense layer with relu activation for the first input
fc1_1 = fullyConnectedLayer(hidden_units, 'Name', 'fc1_1');
re1 = reluLayer('Name', 'relu1');
% one dense layer with linear activation for the second input
fc1_2 = fullyConnectedLayer(hidden_units, 'Name', 'fc1_2');

lgraph = addLayers(lgraph, [fc1_1; re1]);
lgraph = addLayers(lgraph, fc1_2);

% connect created input layer with these two dense layers, respectively
lgraph = connectLayers(lgraph, inputNames{1}, 'fc1_1');
lgraph = connectLayers(lgraph, inputNames{2}, 'fc1_2');

% concat these two branches
concat = depthConcatenationLayer(2, 'Name', 'concat');
lgraph = addLayers(lgraph, concat);
lgraph = connectLayers(lgraph, 'relu1', 'concat/in1');
lgraph = connectLayers(lgraph, 'fc1_2', 'concat/in2');

% create and add layers after concatenation
subLayers = [
    fullyConnectedLayer(hidden_units, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(3, 'Name', 'ready2output')
    reSoftmaxLayer(3, 'regression_softmax')
    regressionLayer('Name', 'output')
];

% connect to the reset layers
lgraph = addLayers(lgraph, subLayers);
lgraph = connectLayers(lgraph, 'concat', 'fc2');

% generate dataset
dataset_regression_generator_multiInputs;
load('../dataset/regression04.mat');

% set training options
opts = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 2, ...
    'L2Regularization', 0.0001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'Verbose', true, ...
    'Plots','training-progress');

% train the network
net = trainNetwork(X, Y, lgraph, opts);

% save training process
saveTrainingProcess('NN06_multiInputs_training_process')