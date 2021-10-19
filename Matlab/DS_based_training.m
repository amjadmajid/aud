clearvars
close all

load label_definitions.mat
%% User input

Training_label = "location";
%% Constructing datastores from user selected data sets

samples_root = "..\Sampled_files\";

Fields = {'general_DS'; 'training_DS'; 'validation_DS';'testing_DS';'metadata'};

Train_locs      = {};
Train_labels    = {};
Val_locs        = {};
Val_labels      = {};
Test_locs       = {};
Test_labels     = {};
sample_metadata = {};
paths           = {};

i = 1;
while true
    [file, path]= uigetfile(samples_root + "*.mat");

    if any(strcmp(path,paths))
        fprintf("dataset already in use\n")
        continue
    else
        paths = cat(1,paths,path);
    end

    if file == 0
        break
    elseif who('-file',fullfile(path, file)) ~= "data"
        fprintf("invalid file structure\n")
    else
        load(fullfile(path, file));
        if ~all(strcmp(fieldnames(data),Fields))
            fprintf("invalid file structure\n")
            continue
        end

        fprintf("%s\n", fullfile(path, file))

        Train_locs =   cat(1, Train_locs,   data.training_DS.Files);
        Train_labels = cat(1, Train_labels, data.training_DS.Labels);
        Val_locs =     cat(1, Val_locs,     data.validation_DS.Files);
        Val_labels =   cat(1, Val_labels,   data.validation_DS.Labels);
        Test_locs =    cat(1, Test_locs,    data.testing_DS.Files);
        Test_labels =  cat(1, Test_labels,  data.testing_DS.Labels);

        sample_metadata{i} = data.metadata;

        clear data
        i = i + 1;
    end
end

if i == 1
    return
end

Train_audio_DS = audioDatastore(Train_locs, 'Labels', Train_labels);
Val_audio_DS   = audioDatastore(Val_locs,   'Labels', Val_labels);
Test_audio_DS  = audioDatastore(Test_locs,  'Labels', Test_labels);

clear Train_locs Train_labels Val_locs Val_labels Test_locs Test_labels

%% shrink the number of data samples for testing
reduced_data = false;
if reduced_data
    Train_audio_DS = splitEachLabel(Train_audio_DS, 0.1, 'randomized','TableVariable',Training_label);
    Val_audio_DS   = splitEachLabel(Val_audio_DS, 0.1, 'randomized','TableVariable',Training_label);
    Test_audio_DS  = splitEachLabel(Test_audio_DS, 0.1, 'randomized','TableVariable',Training_label);
end

%% Prep data stores for training

% remove unused label type
Train_audio_DS.Labels = Train_audio_DS.Labels.(Training_label);
Val_audio_DS.Labels = Val_audio_DS.Labels.(Training_label);
Test_audio_DS.Labels = Test_audio_DS.Labels.(Training_label);

display_distribution(Train_audio_DS, "training DB")
display_distribution(Val_audio_DS, "validation DB")
display_distribution(Test_audio_DS, "Test DB")

% Convert to TransformDatastore object with the transform function for
% forming the proper output format of the read function (as needed for the
% trainNetwork function.

Train_DS = transform(Train_audio_DS, @preprocessForTraining, 'IncludeInfo', true);
Val_DS = transform(Val_audio_DS, @preprocessForTraining, 'IncludeInfo', true);
Test_DS = transform(Test_audio_DS, @preprocessForTraining, 'IncludeInfo', true);

clear Train_audio_DS Val_audio_DS Test_audio_DS
%% Define neural Network architecture

sample = read(Test_DS);
reset(Test_DS);
Input_layer_size = [size(sample{1}),1];
Output_layer_size = length(categories(sample{2}));


layers = [
    imageInputLayer(Input_layer_size,"Name","Input","Normalization","none")

    %dropoutLayer(0.4)
    convolution2dLayer([6 15],32)
    reluLayer()

    convolution2dLayer([3 3], 32)
    reluLayer

    %dropoutLayer(0.4)
    fullyConnectedLayer(64)
    reluLayer()

    %dropoutLayer(0.2)
    fullyConnectedLayer(Output_layer_size)
    softmaxLayer()
    classificationLayer()];

training_options = trainingOptions('adam', ...
    'MaxEpochs',15, ...
    'MiniBatchSize', 16, ...
    'L2Regularization', 0.01, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ValidationData',Val_DS, ...
    'ValidationFrequency', 800, ...
    'ValidationPatience', 7, ...
    'OutputNetwork', 'best-validation-loss');
%% Training

net = trainNetwork(Train_DS, layers, training_options);


%% save network
Save_folder = "Trained_nets";

date = string(datetime('today','Format','dd_MM_yyyy'));

name = "trained_net_" + Training_label + "_" + date;

if reduced_data
    name = name + "_reduced_data";
end

%check for existing files with the same name and add increment to create
%unique names
if exist(Save_folder + name+".mat", 'file')
    i = 1;
    while exist(sprintf("%s(%d).mat",name,i),'file')
        i = i +1;
    end
    name = sprintf("%s(%d)",name,i);
end

save(fullfile(Save_folder, name), 'net', 'training_options', 'Train_DS', 'Val_DS', 'Test_DS', 'sample_metadata')

%% Save training porcess immage
Save_folder = "Training_process_images";
currentfig = findall(groot,'Type','Figure');
savefig(currentfig,fullfile(Save_folder,name))

%% Turn off the pc at if training ends at nigth time
c = fix(clock);
h = c(4);
if h >= 2 && h < 8
    system('shutdown -s')
end

%% Helper functions
function [dataOut,info] = preprocessForTraining(data,info)

%loop around
if true
    data = toroidal_padding(data, 0,0,2,3);
end

dataOut = {data',info.Label};

end

function Z = toroidal_padding(X, t, b, l, r)

inputSize = size(X);

persistent row_copy col_copy;
if isempty(row_copy)
    row_copy = loop_eye(inputSize(1), t, b);
end
if isempty(col_copy)
    col_copy = loop_eye(inputSize(2), l, r)';
end

Z = row_copy * X * col_copy;

end

function Y = loop_eye(base, t, b)

Y = zeros(base + [t+b,0]);
j = mod(-t, base);

for i = 1:size(Y,1)
    j = j +1;
    if j > base
        j = 1;
    end
    Y(i,j) = 1;
end

end