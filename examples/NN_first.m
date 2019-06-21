%% Create a simple classification neural network using
% deep learning toolbox
% NOTE: the compatibality problem of initializer of the layers
%       before R2019a, the fullyConnectedLayer is initilized by normal
%       distribution with zero mean and 0.01 variance by default regardless
%       of the input and output size of the layer. This is changed since
%       R2019a, which is replaced by 'xavier' initializer [1]
%       [1]. https://ww2.mathworks.cn/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html
% The 'xavier' initializer helps stablize training.

% create serveral dense (fullyConnectedLayer) layers
fc1 = fullyConnectedLayer(64);
% fc1.Weights = randn([64, 2])*0.1;
fc2 = fullyConnectedLayer(64);
% fc2.Weights = randn([64, 64])*0.1;
fc3 = fullyConnectedLayer(2);
% fc3.Weights = randn([2, 64])*0.1;

% construct the network by stacking layers
layers = [
    imageInputLayer([2, 1, 1])
    fc1
    reluLayer()
    fc2
    reluLayer()
    fc3
    softmaxLayer()
    classificationLayer()
];

% set rand seed for reproduction
rng(1)
% generate training data set, which has two features, if the average of
% these two features is lower than 0.5, the the label is set as true,
% otherwise set as false
X = rand(2, 1, 1, 10000);
Y = mean(reshape(X, 2, 10000), 1) < 0.5;
Y = reshape(Y, numel(Y), 1);
Y = categorical(Y, [false, true], {'false', 'true'});

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