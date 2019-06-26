function [lgraph, inputNames] = multiInputsLayer(ndims, varargin)
% ndims: row vector indicates number of features for each input layer,
%        e.g., [2, 3] denotes two input layers, which has 2 and 3 features,
%        respectively. 
% varargin: specify the multiple input layers' name, the length of
% varargin{1} must consistent with wth length of ndims
% ref: https://www.mathworks.com/matlabcentral/answers/369328-how-to-use-multiple-input-layers-in-dag-net-as-shown-in-the-figure#comment_700234

inputNames = {};
if nargin > 1
   % check variable
   assert(length(varargin{1})==length(ndims), ...
            'Length of input layers'' names should be the same with ndims.')
   inputLayerNames = varargin{1};
   inputNames = inputLayerNames;
end
if nargin > 2
    error('Please concate the input layers'' names into one cell.')
end

% if only one dim is specified, then just return an imageInputLayer
if length(ndims) == 1
   warning('Only one layer is specified, an imageInputLayer is given.');
   if nargin > 1
       name = varargin{1};
   else
       name = 'input';
   end
   input = imageInputLayer([1, 1, ndims], 'Name', name);
   lgraph = input;
   return;
end

lgraph = layerGraph; % initial the DAG network

% create the input layer with all features concated together along the
% channel dimension (3rd dimension) first
input = imageInputLayer([1, 1, sum(ndims)], 'Name', 'input');
lgraph = addLayers(lgraph, input);

% generate intermediate layers
% the setting keeps the intermediate layer unchanged during the training
settings = '''WeightLearnRateFactor'',0,''BiasLearnRateFactor'',0,''WeightL2Factor'',0,''BiasL2Factor'',0';
for k = 1 : length(ndims)
    for i = 1 : ndims(k)
        if ndims(k) == 1 && nargin > 1
            name = inputLayerNames{k};
        else
            name = sprintf('interLayer_%d_%d', k, i);
        end
        eval(sprintf('%s = convolution2dLayer(1,1,''Name'', ''%s'', %s);', ...
                     name, name, settings));
        eval(sprintf('%s.Weights=zeros(1,1,sum(ndims),1);', name));
        eval(sprintf('%s.Weights(1,1,%d,1)=1;', ...
                     name, i+sum(ndims(1:k-1))) );
        eval(sprintf('%s.Bias=zeros(1,1,1,1);', name));
        eval(sprintf('lgraph=addLayers(lgraph, %s);', name));
        eval(sprintf('lgraph=connectLayers(lgraph,''input'',''%s'');', name));
    end
end

% create concatenation layers and concate intermediate layers according to ndims
for k = 1 : length(ndims)
    if ndims(k) > 1
        if nargin > 1
            concat_name = inputLayerNames{k};
        else
            concat_name = sprintf('input_%d', k);
            inputNames = {inputNames, concat_name};
        end
        eval(sprintf('%s = depthConcatenationLayer(%d, ''Name'', ''%s'');', ...
                     concat_name, ndims(k), concat_name));
        eval(sprintf('lgraph = addLayers(lgraph, %s);', concat_name));
        for i = 1 : ndims(k)
            name = sprintf('interLayer_%d_%d', k, i);
            eval(sprintf('lgraph = connectLayers(lgraph, ''%s'', ''%s/in%d'');', ...
                         name, concat_name, i));
        end
    end
end