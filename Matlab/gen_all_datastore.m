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

label_to_split = "direction";
%% Create general datastores


Top_types = dir(samples_root);
Top_types(~[Top_types.isdir]) = []; %Removes non folders
Top_types(ismember({Top_types.name}, {'.','..'})) = []; %Revomves . and ..

for i = 1:length(Top_types)
    Top_type = Top_types(i).name;
    Top_path = fullfile(samples_root, Top_type);
    
    LoS_types = dir(Top_path);
    LoS_types(~[LoS_types.isdir]) = []; %Removes non folders
    LoS_types(ismember({LoS_types.name}, {'.','..'})) = []; %Revomves . and ..
    
    for j = 1:length(LoS_types)
        LoS_type = LoS_types(j).name;
        LoS_path = fullfile(Top_path, LoS_type);
        
        Sound_types = dir(LoS_path);
        Sound_types(~[Sound_types.isdir]) = []; %Removes non folders
        Sound_types(ismember({Sound_types.name}, {'.','..'})) = []; %Revomves . and ..
        
        for k = 1:length(Sound_types)
            
            Sound_type = Sound_types(k).name;
            Sound_path = fullfile(LoS_path, Sound_type);
            
            Sample_file = "Samples_0s5";
            sample_path = fullfile(Sound_path,Sample_file);
            
            if isfolder(fullfile(sample_path,"datastores"))
                fprintf("already done: %s\n",sample_path)
                continue
            end
            
            %% Create general datastore
            
            audio_DS = audioDatastore(sample_path, 'FileExtensions', '.wav');
            
            %adding label definitions
            load ("label_definitions.mat")
            lss = labeledSignalSet;
            addLabelDefinitions(lss, labelDef);
            
            addMembers(lss,audio_DS)
            
            % Give the labels the proper values
            
            msg = sprintf("Processing\n%s\n%s\n%s\n%s\n",Top_type,LoS_type,Sound_type,Sample_file);
            bar = waitbar(0,msg);
            bar.Children.Title.Interpreter = 'none';
            
            for ii = 1:lss.NumMembers
                
                
                waitbar(ii/lss.NumMembers,bar,sprintf("%s %d/%d",msg,ii,lss.NumMembers))
                
                filename = erase(lss.Source{ii},wildcardPattern+"\");
                DoA = str2double(filename(11:13));   % direction of arival (degrees)
                dist = str2double(filename(5:7));    % distance to source (cm)
                
                setLabelValue(lss,ii,'direction',categorical(DoA))
                setLabelValue(lss,ii,'distance',categorical(dist))
                
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
            
            %% Save datastores
            
            d = split(sample_path,filesep);
            d(1:end-4) = [];
            
            metadata.top_state  = d{1};
            metadata.LoS_state  = d{2};
            metadata.sound_file = d{3};
            d = split(d{4},"_");
            metadata.sample_length = str2double(replace(d{end},"s","."));
            metadata.date = date;
            metadata.splitted_label = label_to_split;
            
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
            
            
        end
    end
    
end



