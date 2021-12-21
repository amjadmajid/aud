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
load ("label_definitions.mat")
%% User inputs

training_partion        = 6;
validation_partition    = 2;
testing_partition       = 2;

total_partitions = training_partion + validation_partition + testing_partition;
training_partion = training_partion/total_partitions;
validation_partition = validation_partition/total_partitions;
testing_partition = testing_partition/total_partitions;

label_to_split = "location";

num_sources = 2*16;
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

        parfor k = 1:length(Sound_types)

            Sound_type = Sound_types(k).name;
            Sound_path = fullfile(LoS_path, Sound_type);

            Sample_file = "Samples";
            sample_path = fullfile(Sound_path,Sample_file);

            if ~isfolder(sample_path)
                continue
            end

            % Skip if there already exists a datastores folder
            if isfolder(fullfile(sample_path,"datastores"))
                fprintf("already done: %s\n",sample_path)

                continue
            end

            %% Create general datastore

            audio_DS = audioDatastore(sample_path, 'FileExtensions', '.wav');
            
            %adding label definitions


            lss = labeledSignalSet;
            addLabelDefinitions(lss, labelDef);

            addMembers(lss,audio_DS)
            source = lss.Source;

            % indices of measurements larger than 250 cm (which will be
            % skipped
            to_remove = zeros(1,lss.NumMembers);

            % Give the labels the proper values


            fprintf("Processing:\n%s\n",sample_path)
            for ii = 1:lss.NumMembers
                if ~strcmp(Top_type,"Noise")
                    filename = erase(source{ii},wildcardPattern+"\");
                    location_data = strsplit(filename,'_');

                    DoA = str2double(location_data{3}(1:3));   % direction of arival (degrees)
                    dist = str2double(location_data{2}(1:3));  % distance to source (cm)
                    loc = strjoin({location_data{3},location_data{2}},'_');

                    % removing distances past 250 cm
                    if dist > 250
                        to_remove(ii) = true;
                        continue
                    end

                    % fixing location name (legacy solution)
                    if ~contains(loc,"deg")
                        loc = insertAfter(loc,3,"deg");
                    end

                    setLabelValue(lss,ii,'direction',categorical(DoA))
                    setLabelValue(lss,ii,'distance',categorical(dist))
                    setLabelValue(lss,ii,'location',categorical({loc}))

                else
                    setLabelValue(lss,ii,'direction',categorical("Noise"))
                    setLabelValue(lss,ii,'distance',categorical("Noise"))
                    setLabelValue(lss,ii,'location',categorical("Noise"))
                end


            end

            removeMembers(lss,find(to_remove)) %remove past 250 cm
            audio_DS = audioDatastore(lss.Source, 'Labels',lss.Labels);
            %% Split datastores


            [training_DS, rem] = splitEachLabel(audio_DS, training_partion, 'randomized','TableVariable',label_to_split);
            [validation_DS, testing_DS] = splitEachLabel(rem, validation_partition/(1-training_partion),'randomized','TableVariable',label_to_split);

            %% Save datastores
            data = [];
            metadata = []

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
                mkdir(sample_path, "datastores_par")
            end
            parsave(fullfile(sample_path,"datastores_par",file_name),data);

            fprintf("Finnished:\n%s\n",sample_path);
        end
    end

end

%% Turn off the pc at nigth time
c = fix(clock);
h = c(4);
if h >= 2 && h < 8
    system('shutdown -s')
end

function parsave(filename, data)
save(filename, "data",'-v7.3')
end

