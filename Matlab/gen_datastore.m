%Script to create an audio datastore object for the audio samples in a
%selected folder. The audio datastore will be split into 3 parts for
%training, validation and testing data for both the distance and direction
%labels seperately.
%The datastores are then transformed to transformed datastore object
%suitable for training.
%The datastores are saved in folders nested in the selected folder.

clearvars
close all
samples_root = "..\Sampled_files\";

%% User inputs

training_partion        = 6;
validation_partition    = 2;
testing_partition       = 2;

sample_path = uigetdir(samples_root);
if sample_path == 0
    return
end

label_to_split = "direction";
%% Create general datastore

audio_DS = audioDatastore(sample_path, 'FileExtensions', '.wav');

%adding label definitions
load ("label_definitions.mat")
lss = labeledSignalSet;
addLabelDefinitions(lss, labelDef);

addMembers(lss,audio_DS)

% Give the labels the proper values
bar = waitbar(0,"Processing:");
for i = 1:lss.NumMembers
    
    msg = sprintf("Processing: %d/%d",i,lss.NumMembers);
    waitbar(i/lss.NumMembers,bar,msg)
    
    filename = erase(lss.Source{i},wildcardPattern+"\");
    DoA = str2double(filename(11:13));   % direction of arival (degrees)
    dist = str2double(filename(5:7));    % distance to source (cm)
    
    setLabelValue(lss,i,'direction',categorical(DoA))
    setLabelValue(lss,i,'distance',categorical(dist))
        
end
close(bar)

audio_DS = audioDatastore(lss.Source, 'Labels',lss.Labels);
%% Split datastores

total_partitions = training_partion + validation_partition + testing_partition;
training_partion = training_partion/total_partitions;
validation_partition = validation_partition/total_partitions;
testing_partition = testing_partition/total_partitions;

[training_DS, rem] = splitDualLabel(audio_DS, label_to_split, training_partion);
[validation_DS, testing_DS] = splitDualLabel(rem, label_to_split, validation_partition/(1-training_partion));

% plot distributions of the labels in data stores 
% display_distribution(audio_DS, "all");
% display_distribution(training_DS, "train");
% display_distribution(validation_DS, "val");
% display_distribution(testing_DS, "test");

%% Save datastores

d = split(sample_path,filesep);
d(1:end-4) = [];

metadata.top_state  = d{1};
metadata.LoS_state  = d{2};
metadata.sound_file = d{3};
d = split(d{4},"_");
metadata.sample_length = str2double(replace(d{end},"s","."));
metadata.date = date;

data.general_DS = audio_DS;
data.training_DS = training_DS;
data.validation_DS = validation_DS;
data.testing_DS = testing_DS;
data.metadata = metadata;


file_name = "DS_" + data.metadata.date;
if ~isfolder(fullfile(sample_path,"datastores")) 
    mkdir(sample_path, "datastores")
end
save(fullfile(sample_path,"datastores",file_name),"data");



