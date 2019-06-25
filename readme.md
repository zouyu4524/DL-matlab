# Deep learning in matlab

Maybe it's more convenient for me to debug deep learning networks in matlab compared with the ones written in Python, such as Tensorflow, Keras or PyTorch. Here we test some functionalitiies related to deep learning (or reinforcement learning) develped in matlab. 

## Prerequest

Matlab release: **R2019a**

## Cheatsheet

**Comparision between Keras and Matlab Deep Learning Toolbox**

|		|Keras		|Matlab 	|
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
|**Loss** |`mse` | `regressionLayer` [^1]|
|		  |`mae` | `maeRegressionLayer` (not officailly included yet) |
| 	 	  |`categorical_crossentropy` | `classificationLayer`[^2] |
|**Optimizer**|`SGD` | `sgdm` |
|             |`RMSprop` | `rmsprop` |
|             |`Adam` | `adam` |

[^1]: Note that, the loss function is binded with the output layer in MATLAB&reg;.
[^2]: A classification layer must be preceded by a softmax layer.