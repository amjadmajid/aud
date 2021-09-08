clearvars

sound = "chirp";
label = "direction";

if sound == "white"
    Sample_path = "..\Sampled_files\Free_Top\Line_of_Sight\W_noise_32s\Samples_2s\";
elseif sound == "chirp"
    Sample_path = "..\Sampled_files\Free_Top\Line_of_Sight\chirps_20_to_20k_in_s314__32s\Samples_2s\";
else
    error("unknown sound")
end
load(Sample_path+"Database.mat")

load label_definitions.mat

%%
% shrink the number of data samples for testing
reduced_data = false;
if reduced_data
    Audio_data_base = splitDualLabel(Audio_data_base, labelDef, 1, 0.3);
end
display_distribution(Audio_data_base,"Total data base")

% split database into training and verification databases
if label == "direction"
    [training_audio_DB, validation_audio_DB] = splitDualLabel(Audio_data_base, labelDef, 1, 2/3);
    
    display_distribution(training_audio_DB, "training DB")
    display_distribution(validation_audio_DB, "validation DB")
    
    % remove lables that are not of interest in training
    training_audio_DB.Labels = training_audio_DB.Labels.direction;
    validation_audio_DB.Labels = validation_audio_DB.Labels.direction;
    
elseif label == "distance"
    [training_audio_DB, validation_audio_DB] = splitDualLabel(Audio_data_base, labelDef, 2, 2/3);
    
    display_distribution(training_audio_DB, "training DB")
    display_distribution(validation_audio_DB, "validation DB")
    
    % remove lables that are not of interest in training
    training_audio_DB.Labels = training_audio_DB.Labels.distance;
    validation_audio_DB.Labels = validation_audio_DB.Labels.distance;
    
else
    error("unknown label")
end

display_distribution(training_audio_DB, "training DB")
display_distribution(validation_audio_DB, "validation DB")

% Convert to TransformDatastore object with the transform function for
% forming the proper output format of the read function (as needed for the
% trainNetwork function.

training_DB = transform(training_audio_DB, @preprocessForTraining, 'IncludeInfo', true);
validation_DB = transform(validation_audio_DB, @preprocessForTraining, 'IncludeInfo', true);

save(Sample_path+"training_"+ label + "_DB.mat", "training_DB")
save(Sample_path+"validation_"+ label + "_DB.mat", "validation_DB")

%% Define neural Network architecture

for i = 1:size(labelDef,1)
    notfound = false;
    if labelDef(i,1).Name == label
        Output_layer_size = size(labelDef(i,1).Categories,1); 
        break
    else 
        notfound = true;
    end
end

if notfound
    error("unknown label")
end

layers = [
    imageInputLayer([11 88200 1],"Name","imageinput","Normalization","none")
    dropoutLayer(0.4,"Name","dropout_2")
    convolution2dLayer([6 15],32,"Name","conv")
    reluLayer("Name","relu") 
    dropoutLayer(0.2,"Name","dropout_1")
    fullyConnectedLayer(Output_layer_size,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options = trainingOptions('adam', ...
    'MaxEpochs',15, ...
    'MiniBatchSize', 10, ...
    'L2Regularization', 0.01, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ValidationData',validation_DB, ...
    'ValidationFrequency', 20);
%% Training
net = trainNetwork(training_DB, layers, options);

% save network
date = string(datetime('today','Format','dd_MM_yyyy'));

name = "trained_net_" + sound + "_" + label + "_" + date;

%check for existing files with the same name and add increment to create
%unique names
if exist(name+".mat", 'file')
    i = 1;
    while exist(sprintf("%s(%d).mat",name,i),'file')
        i = i +1;
    end
    name = sprintf("%s(%d)",name,i);
end

save(name, 'net', 'options','Audio_data_base')

%% Helper functions
function display_distribution(database, plot_title)
% show distribution in database
showing = false;
if ~showing
    return
end
figure

if isa(database.Labels, 'table')
    hist3(double(string(table2array(database.Labels))),'nbins',[12,5])
    xlabel("Direction")
    xticks(0:30:330)
    ylabel("Distance")
    yticks(20:20:100)
else
    histogram(database.Labels)
    if min(double(string(database.Labels))) == 0
        xlabel("Direction")
    else
        xlabel("Distance")
    end
end

title(plot_title)

end

function [dataOut,info] = preprocessForTraining(data,info)

%loop around
if true
    data = toroidal_padding(data, 0,0,2,3);
end

dataOut = {data',info.Label};

end

function Z = toroidal_padding(X, t, b, l, r)

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