clearvars
close all

load trainingData.mat
load validationData.mat

input_size = size(Xtrain);

layers = [
    imageInputLayer([input_size(1:2), 1], "Name","imageinput","Normalization","none")
    convolution2dLayer([3 6],32,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(10,"Name","fc_2")
    reluLayer("Name","relu_4")
    fullyConnectedLayer(12,"Name","fc_1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classification")];

options = trainingOptions('sgdm', ...
    'MaxEpochs',30, ...
    'ValidationData',{Xvalidation,Yvalidation}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(Xtrain,Ytrain,layers,options);

