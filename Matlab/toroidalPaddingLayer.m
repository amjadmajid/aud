classdef toroidalPaddingLayer < nnet.layer.Layer
    % This layer padds a 2D array as if it were wrapped around a toroid
    
    %#codegen
    
    properties
        % Layer properties.
        
        PaddingSize
    end
    
    methods
        function layer = toroidalPaddingLayer(paddingSize, name)
            % paddingSize: Size of padding to apply to input borders,
            % specified as a vector [t b l r] of four nonnegative integers,
            % where t is the padding applied to the top, b is the padding
            % applied to the bottom, l is the padding applied to the left,
            % and r is the padding applied to the right. r
            
            if ~all(size(paddingSize) == [1,4])
                error("padding size must be of size [1 , 4]")
            else
                %check that all inputs are intergers and >= 0 (modified
                %from https://nl.mathworks.com/matlabcentral/answers/377094-how-to-check-the-input-from-user-is-positive-integer-number#answer_300082 )
                int_ge_0 = @(n) (rem(n,1) == 0) & (n >= 0);
                if ~all(int_ge_0(paddingSize))
                    error("paddingSize parameters must be non negative intergers")
                end
            end
            
            layer.Name = name;
            layer.Description = "Toroidal padding with paddingSize: [" + num2str(paddingSize) + "]";
            layer.PaddingSize = paddingSize;
        end
        
        function Z = predict(layer, X)
            t = layer.PaddingSize(1);
            b = layer.PaddingSize(2);
            l = layer.PaddingSize(3);
            r = layer.PaddingSize(4);
            
            inputSize = size(X);
            outputSize = inputSize;
            outputSize(1) = outputSize(1) + t + b;
            outputSize(2) = outputSize(2) + l + r;
            
            Z = zeros(outputSize, 'like', X);
            
            for i = 1:outputSize(1)
                Xi = mod(i-1-t,inputSize(1))+1;
                for j = 1:outputSize(2)
                    Xj = mod(j-1-l,inputSize(2))+1;
                    
                    Z(i,j,:) = X(Xi,Xj,:);
                end
                
            end
        end
        
        %         function [dLdX1, …, dLdXn, dLdW1, …, dLdWk] = ...
        %                 backward(layer, X1, …, Xn, Z1, …, Zm, dLdZ1, …, dLdZm, memory)
        %             % (Optional) Backward propagate the derivative of the loss
        %             % function through the layer.
        %             %
        %             % Inputs:
        %             %         layer             - Layer to backward propagate through
        %             %         X1, ..., Xn       - Input data
        %             %         Z1, ..., Zm       - Outputs of layer forward function
        %             %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
        %             %         memory            - Memory value from forward function
        %             % Outputs:
        %             %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
        %             %                             inputs
        %             %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
        %             %                             learnable parameter
        %
        %             % Layer backward function goes here.
        %         end
    end
end