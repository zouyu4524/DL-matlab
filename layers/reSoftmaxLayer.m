classdef reSoftmaxLayer < nnet.layer.Layer
    % Custom softmax layer for regression task, the output of previous
    % layer will be conducted the softmax operation along the dim specified
    % by `dim`.
    %   [Input] dim: specify dim to operate softmax
    %   [Input] name: specify name of the layer

    properties
       Dim % dim to operate softmax
    end
    
    methods
        function layer = reSoftmaxLayer(dim, name) 
            % layer = ReSoftmaxLayer(name) creates a softmax activation layer
            % and specifies the layer name.

            % specify dim to operate softmax
            layer.Dim = dim;
            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "softmax activation";

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            X = X - max(X,[],layer.Dim);
            X = exp(X);
            Z = X./sum(X,layer.Dim);
        end
        
        function [dLdX] = backward(layer, ~, Z, dLdZ, ~)
            % [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % backward propagates the derivative of the loss function
            % through the layer.
            % Inputs:
            %         layer    - Layer to backward propagate through
            %         X        - Input data
            %         dLdZ     - Gradient propagated from the deeper layer
            % Outputs:
            %         dLdX     - Derivative of the loss with respect to the
            %                    input data
            Z = nnet.internal.cnn.util.boundAwayFromZero(Z);
            dotProduct = sum(Z.*dLdZ, layer.Dim);
            dLdX = dLdZ - dotProduct;
            dLdX = dLdX.*Z;

        end
    end
end