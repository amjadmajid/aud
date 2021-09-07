clearvars
X(:,:,1,1) = magic(3);
X(:,:,2,1) = eye(3);
X(:,:,1,2) = ones(3,3);
X(:,:,2,2) = zeros(3,3);

paddingSize = [1 2 1 3];
t = paddingSize(1);
b = paddingSize(2);
l = paddingSize(3);
r = paddingSize(4);


inputSize = size(X);
outputSize = inputSize;
outputSize(1) = outputSize(1) + t + b;
outputSize(2) = outputSize(2) + l + r;

a = zeros(outputSize, 'like', X);
%a(1+t:t+inputSize(1),1+l:l+inputSize(2),:) = X

Z = zeros(outputSize, 'like', X);

for i = 1:outputSize(1)
    Xi = mod(i-1-t,inputSize(1))+1;
    for j = 1:outputSize(2)
        Xj = mod(j-1-l,inputSize(2))+1;
    
        Z(i,j,:) = X(Xi,Xj,:);
    end
    
end

disp(Z)