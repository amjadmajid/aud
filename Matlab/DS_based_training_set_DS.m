clearvars
close all

%load label_definitions.mat
%% User input

Training_label = "location";

sample_type = "0s024";

Reload_last_DS = true;

added_noise = true;

if ~any(strcmp(sample_type,["0s024", "0s048"]))
    error("non valid sample type")
end
%% Constructing datastores from user selected data sets

if Reload_last_DS && isfile("DS_last_training.mat")
    fprintf("reloding dataset from last training session\n")
    load("DS_last_training.mat")
    fprintf("done loading\n")
else
    Fields = {'general_DS'; 'training_DS'; 'validation_DS';'testing_DS';'metadata'};
    Train_locs      = {};
    Train_labels    = {};
    Val_locs        = {};
    Val_labels      = {};
    Test_locs       = {};
    Test_labels     = {};
    sample_metadata = {};

    LoS_type = ["Line_of_Sight", "Non_Line_of_Sight"];
    for i = 1:2
        for j = 0:7
            DS_loc = fullfile("..","Sampled_files","Obstructed_Top",LoS_type(i), ...
                sprintf("chirp_train_chirp_%s_%d",sample_type,j),"Samples","datastores");
            DS_file = dir(DS_loc + '\DS_*.mat');
            DS_file = DS_file(end).name;

            fprintf("loading: %s\n",fullfile(DS_loc,DS_file));

            load(fullfile(DS_loc,DS_file));
            if ~all(strcmp(fieldnames(data),Fields))
                error("invalid file structure in file %s\n",fullfile(DS_loc,DS_file))
            end

            Train_locs =   cat(1, Train_locs,   data.training_DS.Files);
            Train_labels = cat(1, Train_labels, data.training_DS.Labels);
            Val_locs =     cat(1, Val_locs,     data.validation_DS.Files);
            Val_labels =   cat(1, Val_labels,   data.validation_DS.Labels);
            Test_locs =    cat(1, Test_locs,    data.testing_DS.Files);
            Test_labels =  cat(1, Test_labels,  data.testing_DS.Labels);

            sample_metadata{end+1} = data.metadata;

            clear data
        end
    end

    fprintf("Done loading\n\n" + ...
        "Generate datastores\n")

    Train_audio_DS = audioDatastore(Train_locs, 'Labels', Train_labels);
    Val_audio_DS   = audioDatastore(Val_locs,   'Labels', Val_labels);
    Test_audio_DS  = audioDatastore(Test_locs,  'Labels', Test_labels);

    clear Train_locs Train_labels Val_locs Val_labels Test_locs Test_labels

    fprintf("done generating:\n" + ...
        "\tnum training samples: %d\n" + ...
        "\tnum validation samples: %d\n" + ...
        "\tnum testing samples: %d\n" + ...
        "\tTotal samples: %d\n", ...
        length(Train_audio_DS.Files), length(Val_audio_DS.Files), length(Test_audio_DS.Files), ...
        length(Train_audio_DS.Files) +length(Val_audio_DS.Files) +length(Test_audio_DS.Files))
    %% shrink the number of data samples for testing
    reduced_data = false;
    if reduced_data
        Train_audio_DS = splitEachLabel(Train_audio_DS, 0.1, 'randomized','TableVariable',Training_label);
        Val_audio_DS   = splitEachLabel(Val_audio_DS, 0.1, 'randomized','TableVariable',Training_label);
        Test_audio_DS  = splitEachLabel(Test_audio_DS, 0.1, 'randomized','TableVariable',Training_label);
    end

    save("DS_last_training","Train_audio_DS","Val_audio_DS","Test_audio_DS","sample_metadata","reduced_data")
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
if added_noise
    Train_DS = transform(Train_audio_DS, @preprocessForTraining_noise, 'IncludeInfo', true);
    Val_DS = transform(Val_audio_DS, @preprocessForTraining_noise, 'IncludeInfo', true);
    Test_DS = transform(Test_audio_DS, @preprocessForTraining_noise, 'IncludeInfo', true);
else
    Train_DS = transform(Train_audio_DS, @preprocessForTraining, 'IncludeInfo', true);
    Val_DS = transform(Val_audio_DS, @preprocessForTraining, 'IncludeInfo', true);
    Test_DS = transform(Test_audio_DS, @preprocessForTraining, 'IncludeInfo', true);
end
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

    %     convolution2dLayer([3 3], 32)
    %     reluLayer
    %
    %     %dropoutLayer(0.4)
    %     fullyConnectedLayer(64)
    %     reluLayer()

    fullyConnectedLayer(64)
    reluLayer()

    %dropoutLayer(0.2)
    fullyConnectedLayer(Output_layer_size)
    softmaxLayer()
    classificationLayer()];

MiniBatchSize = 128;
ValidationFrequency = floor(length(Train_DS.UnderlyingDatastores{1}.Files)/MiniBatchSize /2); % have two validation steps per epoch

training_options = trainingOptions('adam', ...
    'MaxEpochs',15, ...
    'MiniBatchSize', MiniBatchSize, ...
    'L2Regularization', 0.005, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ValidationData',Val_DS, ...
    'ValidationFrequency', ValidationFrequency, ...
    'ValidationPatience', 5, ...
    'OutputNetwork', 'best-validation-loss');
%% Training

net = trainNetwork(Train_DS, layers, training_options);

fprintf("training complete")

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
mkdir(fullfile(Save_folder, name));

%save figure
currentfig = findall(groot,'Type','Figure');
savefig(currentfig,fullfile(Save_folder,name,"training_process_figure"))

%extract plot data from figure
acc_plot_axis = currentfig.Children.Children.Children(1).Children(2).Children(1).Children;
loss_plot_axis = currentfig.Children.Children.Children(1).Children(2).Children(2).Children;

itteration = acc_plot_axis.Children(6).XData';
acc_raw = acc_plot_axis.Children(6).YData';
acc_smoothed = acc_plot_axis.Children(5).YData';
loss_raw = loss_plot_axis.Children(6).YData';
loss_smoothed = loss_plot_axis.Children(5).YData';

training_progress.training = table(itteration,acc_raw,acc_smoothed,loss_raw,loss_smoothed);

itteration = acc_plot_axis.Children(4).XData';
acc = acc_plot_axis.Children(4).YData';
loss = loss_plot_axis.Children(4).YData';
    
training_progress.validation = table(itteration,acc,loss);

training_progress.final.sample = acc_plot_axis.Children(2).XData;
training_progress.final.acc = acc_plot_axis.Children(2).YData;
training_progress.final.loss = loss_plot_axis.Children(2).YData;

%save extracted plot data
save(fullfile(Save_folder,name,"training_process_data"),'training_progress')

%% Turn off the pc at if training ends at nigth time
c = fix(clock);
h = c(4);
if h >= 2 && h < 8
    system('shutdown -s')
end

%% show confusion matrix and distance error (if it isn't night time)
[val_predict, val_true] = Conf_matrix(net,Val_DS, "validation");

%% Helper functions
function [dataOut,info] = preprocessForTraining(data,info)

%loop around
if true
    data = toroidal_padding(data, 0,0,2,3);
end

dataOut = {data',info.Label};

end

function [dataOut,info] = preprocessForTraining_noise(data,info)    

%add noise
snr = 5;
data = awgn(data,snr,'measured');

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