clearvars

% location of the soundsamples
Sample_path = "..\Sampled_files\Free_Top\Line_of_Sight\chirps_20_to_20k_in_s314__32s\Samples_2s\";

%load labels
load label_definitions.mat

% Add labels to dataset
lss = labeledSignalSet;
addLabelDefinitions(lss, labelDef)

% Add audio sources to the dataste
src = audioDatastore(Sample_path,'FileExtensions','.wav');
addMembers(lss,src)

% Give the labels the proper values
for i = 1:lss.NumMembers
    
    filename = erase(lss.Source{i},wildcardPattern+"\");
    DoA = str2double(filename(11:13));   % direction of arival (degrees)
    dist = str2double(filename(5:7));    % distance to source (cm)
    
    setLabelValue(lss,i,'direction',categorical(DoA))
    setLabelValue(lss,i,'distance',categorical(dist))

end

%Store labeled audio database
Audio_data_base = audioDatastore(lss.Source, 'Labels',lss.Labels);
save(Sample_path+"Database.mat","Audio_data_base")