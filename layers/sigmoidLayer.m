classdef sigmoidLayer < nnet.layer.Layer
    % Example custom Sigmoid layer.

    methods
        function layer = sigmoidLayer(name) 
            % layer = sigmoidLayer(name) creates a sigmoid activation layer
            % and specifies the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Sigmoid activation";

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = 1./(1+exp(-X));
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
            
            dLdX = Z.*(1-Z) .* dLdZ;

        end
    end
end