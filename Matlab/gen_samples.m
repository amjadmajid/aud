% Script for converting recorded signals into samples of fixed length 

clearvars
close all

buffer_length = 1;      %time [s] at the beginning and and of the recordign which will be discarded
sample_length = 0.5;    % Duration of each sample in seconds


recordings_root = "..\Recorded_files\";
samples_root = "..\Sampled_files\";
sample_folder = "Samples_"+num2str(floor(sample_length))+"s";
%preventing decimal points in the folder name
if sample_length - floor(sample_length) > 0
    rem = num2str(sample_length-floor(sample_length),3);
    sample_folder = sample_folder + rem(3:end);
end

Top_types = dir(recordings_root);
Top_types(~[Top_types.isdir]) = []; %Removes non folders
Top_types(ismember({Top_types.name}, {'.','..'})) = []; %Revomves . and ..

for i = 1:length(Top_types)
    Top_type = Top_types(i).name;
    Top_path = Top_type;
    
    LoS_types = dir(recordings_root + Top_path);
    LoS_types(~[LoS_types.isdir]) = []; %Removes non folders
    LoS_types(ismember({LoS_types.name}, {'.','..'})) = []; %Revomves . and ..
    
    for j = 1:length(LoS_types)
        LoS_type = LoS_types(j).name;
        LoS_path = Top_path + "\" + LoS_type;
    
        Sound_types = dir(recordings_root + LoS_path);
        Sound_types(~[Sound_types.isdir]) = []; %Removes non folders
        Sound_types(ismember({Sound_types.name}, {'.','..'})) = []; %Revomves . and ..
        
        for k = 1:length(Sound_types)
            
            Sound_type = Sound_types(k).name;
            Sound_path = LoS_path + "\" + Sound_type +"\";
            
            Recordings = dir(recordings_root + Sound_path+"Raw_recordings\*.wav");
            
            %check if the folder is empty
            if isempty(Recordings)
                continue
            end
            
            % Make folder to store samples
            Storage_path = samples_root + Sound_path + sample_folder;
            if ~isfolder(Storage_path) 
                mkdir(samples_root + Sound_path, sample_folder)
            end
            
            % use parallel pool to go over all recordings in the folder
            parfor ii = 1:length(Recordings)
                Recording = Recordings(ii).name;
                fprintf("now processing: %s\n",Recording)
                
                % test if this recording has been sampled before and skip
                % the recording if it is the case
                if  ~isempty(dir(Storage_path+'\'+Recording(1:end-4)+'.wav'))
                    continue
                end
                
                Rec_info = audioinfo(Recordings(ii).folder + "\" + Recording);
                
                % Remove buffer at the beginning and end of the recording
                Samples = [1+Rec_info.SampleRate*buffer_length, ...
                    Rec_info.TotalSamples - Rec_info.SampleRate*buffer_length];
                [y, ~] = audioread(Recordings(ii).folder + "\" + Recording, Samples);
                
                % Cut recording into samples of [Sample_length] seconds
                jj = 0;
                while size(y,1) >= Rec_info.SampleRate*sample_length 
                    y_sample = y(1:Rec_info.SampleRate*sample_length,:);
                    y(1:Rec_info.SampleRate*sample_length,:) = [];
                    audiowrite(sprintf('%s\\%s_s%03d.wav',Storage_path,Recording(1:end-4),jj),...
                        y_sample,Rec_info.SampleRate);
                    jj = jj+1;
                end     
            end
        end
    end
    
end