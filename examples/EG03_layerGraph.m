%% Create a simple classification neural network using
% deep learning toolbox. `layerGraph` is used to build the network
% NOTE: the compatibality problem of initializer of the layers
%       before R2019a, the fullyConnectedLayer is initilized by normal
%       distribution with zero mean and 0.01 variance by default regardless
%       of the input and output size of the layer. This is changed since
%       R2019a, which is replaced by 'xavier' initializer [1]
%       [1]. https://ww2.mathworks.cn/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html
% The 'xavier' initializer helps stablize training.

% deep learning toolbox provides two methods to build a neural network,
% i.e., 1) stack layers into an array, 2) or using `layerGraph` method. The
% first method outputs a SeriesNetwork and the second one outputs
% DAGNetwork. The second one is more flexible compared to the first one.

% add path
addpath('../utils')

%% Create layers with name specified
% input layer, note that `name` must be specified for each layer when using
% layerGraph to build the network. This can be omitted when creating a
% SeriesNetwork.
input = imageInputLayer([1, 1, 2], 'Name', 'input');

% create serveral dense (fullyConnectedLayer) layers
fc1 = fullyConnectedLayer(64, 'Name', 'fc1');
fc2 = fullyConnectedLayer(64, 'Name', 'fc2');
fc3 = fullyConnectedLayer(2, 'Name', 'ready2output');

% create activation layers
re1 = reluLayer('Name', 'relu1');
re2 = reluLayer('Name', 'relu2');
softmax = softmaxLayer('Name', 'softmax');

% create output layer
output = classificationLayer('Name', 'output');

% initialize a layerGraph
lgraph = layerGraph();

%% 'Silly' way to add and connect layers
% add layers into lgraph
lgraph = addLayers(lgraph, input);
lgraph = addLayers(lgraph, fc1);
lgraph = addLayers(lgraph, fc2);
lgraph = addLayers(lgraph, fc3);
lgraph = addLayers(lgraph, re1);
lgraph = addLayers(lgraph, re2);
lgraph = addLayers(lgraph, softmax);
lgraph = addLayers(lgraph, output);

% connect layers
lgraph = connectLayers(lgraph, 'input', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'relu1');
lgraph = connectLayers(lgraph, 'relu1', 'fc2');
lgraph = connectLayers(lgraph, 'fc2', 'relu2');
lgraph = connectLayers(lgraph, 'relu2', 'ready2output');
lgraph = connectLayers(lgraph, 'ready2output', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'output');

%% Practical NOTICE
% The above operations are redundant and just for an illustration for how
% to use layerGraph and corresponding `addLayers`, `connectLayers`
% functions. Actually, we can use addLayers in the following way, which is
% more practical during the development. The addLayers will automatically
% connects the layers in the array sequentially.

% Please comment the following codes out and comment the codes in above
% section to test it.
% layers = [
%     input
%     fc1
%     re1
%     fc2
%     re2
%     fc3
%     softmax
%     output
% ];
% lgraph = addLayers(lgraph, layers);

%% The rest codes
% plot lgraph, show structure of the network
plot(lgraph)

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

% train the network, here we use lgraph to represents the network instead
% of layers in example 01/02.
net = trainNetwork(X, Y, lgraph, opts);

% save training process
saveTrainingProcess('NN03_layerGraph_training_process')