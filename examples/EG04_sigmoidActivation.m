%% Create a simple regression neural network using
% deep learning toolbox and a self-defined sigmoid activation layer
% NOTE: the compatibality problem of initializer of the layers
%       before R2019a, the fullyConnectedLayer is initilized by normal
%       distribution with zero mean and 0.01 variance by default regardless
%       of the input and output size of the layer. This is changed since
%       R2019a, which is replaced by 'xavier' initializer [1]
%       [1]. https://ww2.mathworks.cn/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html
% The 'xavier' initializer helps stablize training.

% The built-in deep learning toolbox doesn't provide sigmoid activation for
% now (R2019a), we can define it by ourself according to the following tutorial.
% ref: https://www.mathworks.com/help/deeplearning/ug/define-custom-deep-learning-layer.html
% Please find details in sigmoidLayer under folder layers/.

% sigmoid function rectifies inputs in range [0,1], while tanh rectifies
% inputs to [-1, 1]. We can choose different activations by demand.

% add path
addpath('../utils')
addpath('../layers')

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
    % we use sigmoid activation here instead of tanh activation
    sigmoidLayer('sigmoid_activation')
    output
];

% generate dataset
dataset_regression_generator_sigmoid;
load('../dataset/regression02.mat');

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
saveTrainingProcess('NN04_sigmoid_training_process')