# Deep learning in MATLAB  

It's more convenient for me to debug deep learning networks in MATLAB compared with the ones written in Python, such as Tensorflow, Keras or PyTorch. Here we test some functionalitiies related to deep learning develped in MATLAB, i.e., deep learning toolbox.

- [Deep learning in MATLAB](#deep-learning-in-matlab)
  * [Prerequest](#prerequest)
  * [Demo](#demo)
  * [Example](#example)
  * [Dataset](#dataset)
  * [Build a neural network](#build-a-neural-network)
  * [Pros and Cons](#pros-and-cons)
  * [Cheatsheet](#cheatsheet)
  * [Reference](#reference)

## Prerequest

Matlab release: **R2019a**

The deep learning toolbox is introduced since R2016a. However, it's better to use the version greater than R2018b due to quite a lot [updates](https://ww2.mathworks.cn/help/deeplearning/release-notes.html) on the toolbox.

## Demo

A simple neural network is built for a classification task. The data set is generated randomly with two-feature entries, each feature ranges from 0 to 1. If the average of these two features is lower than 0.5, then the corresponding label is true and false otherwise. We build a network with two fully connected hidden layer (i.e., Dense layer in Keras).

```
% create serveral dense (fullyConnectedLayer) layers
fc1 = fullyConnectedLayer(64);
fc2 = fullyConnectedLayer(64);
fc3 = fullyConnectedLayer(2);

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
Y = reshape(Y, 1, numel(Y));
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
```

## Example

**EG01_classification**: We build a simple neural network (`SeriesNetwork`) with two hidden layers for a classification task.  

**EG02_regression**: We build a neural network (`SeriesNetwork`) with two hidden layers for a regression task.  

**EG03_layerGraph**: We build a `DAGNetwork` using `layerGraph` for a classification task.  

**EG04_sigmoidActivation**: We develop a sigmoid activation layer in this example.  

**EG05_softmaxActivation**: We develop a softmax activation layer for a regression task. (The built-in `softmaxLayer` in the toolbox is designated to the classification layer for now [R2019a].) 

**EG06_multipleInputs**: We develop an input layer with multiple inputs support.  

## Dataset

**Classification**  
The input data is randomly generated with two-feature entries. Each entry's output is labelled according to the criterion `mean(x)<0.5`, i.e., if the average of the two features is less than 0.5, the output is marked as true and false otherwise.  

|		|Size/Type	|Value	|
|:-----:|:--------	|:------|
|	Input 	| `1x1x2 double` | Each feature ranges in [0,1] |
|	Output  | [`categorical`](https://www.mathworks.com/help/matlab/ref/categorical.html)  | `true` or `false` |

**Regression**  
The input data is the same with the one in classification dataset. The output has three dimensions and each one maps the input to a scalar via a function defined as follows:  

|		|Size/Type	|Value	|
|:-----:|:--------	|:------|
|	Input 	| `1x1x2 double` | Each feature ranges in [0,1] |
|	x1      | `double` 		 | Input(1) |
|	x1      | `double` 		 | Input(2) |
|	y1 	    | `double`		 | x1 + x2 - 1 |
|	y2 		| `double` 		 | 2\*x1\*x2 - 1|
|	y3 		| `double`		 | x1^2 + x2^2 - 1|
|   Output  | `1x3 double`   | [y1, y2, y3] |

## Build a neural network

We present a comparison between Keras and Matlab on the process of building a neural network.

|		|MATLAB 		|Keras |
|------:|:--------------|:-----|
|1.		| Stack layers one by one. The `loss` function is binded with the output layer. | Stack layers one by one |
|2. 	| Set training options, including optimizer, learning rate and etc. | Compile the model with optimizer, loss function |
|3. 	| Train the network via `trainiNetwork` | Train the network via `fit` function. |

Basically, the processes of building a network via MATLAB and Keras are similar. It's worth to note three differences:  
1. The loss function is bined with the output layer in MATLAB.  
2. The settings of the training process are specificied via a `trainingOptions` object in MATLAB.  
3. The model compile process is integrated with `trainNetwork` in MATLAB.  

## Pros and Cons

**Pros**  
1. It's convenient to debug the neural network via its powerful built-in plot functions. We can turn on the training plot to moniter the training process via setting `Plots` as `training-progress` in `trainingOptions`.  
2. The `activations` method of the network object can perform forward operation to any layer of the network. It's useful to debug and adjust network structure if it's necessary.  

**Cons**  
1. There is no multiple inputs layer supported for now (R2019a). (There is a [workaround](https://www.mathworks.com/matlabcentral/answers/369328-how-to-use-multiple-input-layers-in-dag-net-as-shown-in-the-figure#comment_700234).)  
2. The training is not as fast as Python.  
3. The layers in MATLAB is not fruitful compared with the framework written in Python. (It provides a [tutorial](https://www.mathworks.com/help/deeplearning/ug/define-custom-deep-learning-layer.html) on implementation of self-defined layers.)  

## Cheatsheet

**Comparision between Keras and Matlab Deep Learning Toolbox**

|		|Keras		|MATLAB 	|
|-------:|:-----------|:-----------|
|**Layer**|`Dense`  |`fullyConnectedLayer`|
|		  |`Input`	|`imageInputLayer` |
|		  |`Flatten`|`flattenLayer` |
|		  |`Concatenate`|`concatenationLayer`|
|		  |`BatchNormalization`|`batchNormalizationLayer`|
|**Activation**|`Activation('relu')`|`reluLayer`|
|	      |`Activation('tanh')`|`tanhLayer`|
|		  |`Activation('linear')`|*nothing after the previous layer*|
|         |`Activation('softmax')`| `softmaxLayer` |
|**Loss** |`mse` | `regressionLayer` <sup>1</sup>|
|		  |`mae` | `maeRegressionLayer` (not officailly included yet) |
| 	 	  |`categorical_crossentropy` | `classificationLayer` <sup>2</sup> |
|**Optimizer**|`SGD` | `sgdm` |
|             |`RMSprop` | `rmsprop` |
|             |`Adam` | `adam` |
|**Method**|`fit`	| `trainNetwork` |
|          |`predict` | `predict` / `classify` |

[1]: Note that, the loss function is binded with the output layer in MATLAB&reg;.  
[2]: A classification layer must be preceded by a softmax layer.

## Reference

1. Mathworks, [Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/index.html)  
2. Mahmoud Afifi, [how to use multiple input layers in DAG net as shown in the figure](https://www.mathworks.com/matlabcentral/answers/369328-how-to-use-multiple-input-layers-in-dag-net-as-shown-in-the-figure#comment_700234)  
3. Mathworks, [Define Custom Deep Learning Layer with Learnable Parameters](https://www.mathworks.com/help/deeplearning/ug/define-custom-deep-learning-layer.html)  
