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