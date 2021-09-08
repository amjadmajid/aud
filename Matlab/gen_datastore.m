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
%% Create general datastore

audio_DS = audioDatastore(sample_path, 'FileExtensions', '.wav');

%adding tlabel definitions
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

