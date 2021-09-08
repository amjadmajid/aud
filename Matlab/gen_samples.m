% Chops the larger audio files into 2 second samples

clearvars
close all

Buffer_seconds = 1; % the number of seconds that will be removed from the start and end of the recording
Sample_length = 2;  % the length of a sample [s]

Recordings_root = "..\Recorded_files\";
Samples_root = "..\Sampled_files\";
Sample_folder = "Samples_"+num2str(Sample_length)+"s";

Top_types = dir(Recordings_root);
Top_types(~[Top_types.isdir]) = []; %Removes non folders
Top_types(ismember({Top_types.name}, {'.','..'})) = []; %Revomves . and ..

for i = 1:length(Top_types)
    Top_type = Top_types(i).name;
    Top_path = Top_type;
    
    LoS_types = dir(Recordings_root + Top_path);
    LoS_types(~[LoS_types.isdir]) = []; %Removes non folders
    LoS_types(ismember({LoS_types.name}, {'.','..'})) = []; %Revomves . and ..
    
    for j = 1:length(LoS_types)
        LoS_type = LoS_types(j).name;
        LoS_path = Top_path + "\" + LoS_type;
    
        Sound_types = dir(Recordings_root + LoS_path);
        Sound_types(~[Sound_types.isdir]) = []; %Removes non folders
        Sound_types(ismember({Sound_types.name}, {'.','..'})) = []; %Revomves . and ..
        
        for k = 1:length(Sound_types)
            
            Sound_type = Sound_types(k).name
            Sound_path = LoS_path + "\" + Sound_type +"\";
            
            % Make folder to store samples
            Storage_path = Samples_root + Sound_path + Sample_folder;
            if ~isfolder(Storage_path) 
                mkdir(Samples_root + Sound_path, Sample_folder)
            end
            
    
            Recordings = dir(Recordings_root + Sound_path+"Raw_recordings\*.wav");
            parfor ii = 1:length(Recordings)
                Recording = Recordings(ii).name;
                Rec_info = audioinfo(Recordings(ii).folder + "\" + Recording);
                
                % Remove buffer at the beginning and end of the recording
                Samples = [1+Rec_info.SampleRate*Buffer_seconds, ...
                    Rec_info.TotalSamples - Rec_info.SampleRate*Buffer_seconds];
                [y, ~] = audioread(Recordings(ii).folder + "\" + Recording, Samples);
                
                % Cut recording into samples of [Sample_length] seconds
                jj = 0;
                while size(y,1) >= Rec_info.SampleRate*Sample_length 
                    y_sample = y(1:Rec_info.SampleRate*Sample_length,:);
                    y(1:Rec_info.SampleRate*Sample_length,:) = [];
                    audiowrite(sprintf('%s\\%s_s%03d.wav',Storage_path,Recording(1:end-4),jj),...
                        y_sample,Rec_info.SampleRate);
                    jj = jj+1;
                end     
            end
        end
    end
    
end