function multi_source_samp_gen(LoS_data, NLoS_data, LoS_sum, NLoS_sum,...
    num_sources, paritions, storage_root, num_samples, ...
    base_start, audio_length, shift_range)


if ~isfolder(storage_root)
    mkdir(storage_root)
end

fields = fieldnames(LoS_data);

all_labels = categories(LoS_data(1).(fields{1}).Labels.location(1));
all_labels(strcmp(all_labels, 'Noise')) = [];
all_labels = strjoin(all_labels,'-');

%pool = gcp;

multiWaitbar('fields',0);
for j = 1:length(fields)
    field = fields{j};

    destination = fullfile(storage_root,field);
    if ~isfolder(destination)
        mkdir(destination)
    end

%     for i = 1:pool.NumWorkers
%         filename = sprintf("_ledger_%d.txt",i);
%         txt_idx = fopen(fullfile(destination,filename),'wt+');
%         fprintf(txt_idx,"idx \t sources\n");
%         fclose(txt_idx);
%     end

    NLoS_threshold = NLoS_sum.(field)/(NLoS_sum.(field) + LoS_sum.(field));

    required_samples = floor(num_samples * paritions.(field));
    parfor i = 1: required_samples
        chirps = randperm(8,num_sources);

        labels = cell(num_sources,1);
        sample_names = cell(num_sources,1);

        % make base

        %determine if the sound comes from the nlos or the los data set
        if rand < NLoS_threshold
            file = datasample(NLoS_data(chirps(1)).(field).Files,1);
        else
            file = datasample(LoS_data(chirps(1)).(field).Files,1);
        end

        file = file{1};

        % determine name part

        % location data
        location = strsplit(file,'\');
        location = location{end}(1:end-4);
        location = erase(location, 'rec_');

        % chirp data
        chirp = strsplit(file,'\');
        chirp = chirp{end-2}(end);

        % determine label
        label = strsplit(location,'_');
        label = flip(label(1:2));

        if ~contains(label{1},'deg')
            label{1} = strcat(label{1},'deg');
        end

        sample_names{1} = strjoin({chirp,location},'_');
        labels{1} = strjoin(label,'_');

        %read audio
        [audio, Fs] = audioread(file);
        %crop audio
        crop_range = floor(base_start*Fs) + (0 : ceil(audio_length*Fs));
        audio = audio(crop_range,:);


        %superimpose other samples onto base
        for k = 2:num_sources
            while true
                %determine if the sound comes from the nlos or the los data set
                if rand < NLoS_threshold
                    file = datasample(NLoS_data(chirps(k)).(field).Files,1);
                else
                    file = datasample(LoS_data(chirps(k)).(field).Files,1);
                end
                file = file{1};

                % determine name part

                % location data
                location = strsplit(file,'\');
                location = location{end}(1:end-4);
                location = erase(location, 'rec_');

                % chirp data
                chirp = strsplit(file,'\');
                chirp = chirp{end-2}(end);

                % determine label
                label = strsplit(location,'_');
                label = flip(label(1:2));

                if ~contains(label{1},'deg')
                    label{1} = strcat(label{1},'deg');
                end
                label = strjoin(label,'_');

                %only continue if next sample is from another location 
                if ~contains(labels(1:k-1), label)
                    break
                end
            end

            sample_names{k} = strjoin({chirp,location},'_');
            labels{k} = label;

            %read audio
            sample_audio = audioread(file);
            %crop audio
            delay = shift_range(1) + diff(shift_range) *rand;
            crop_range = floor((base_start-delay)*Fs) + (0 : ceil(audio_length*Fs));
            sample_audio = sample_audio(crop_range,:);


            %superimpose audio
            audio = audio + sample_audio;

        end

        label = strjoin(labels,'-');

        filename = sprintf('rec_%d[%s.wav',num_sources,strjoin(sample_names,'['));

%         ledger_file = sprintf("_ledger_%d.txt",get(getCurrentTask,'ID'));
%         txt_idx = fopen(fullfile(destination,ledger_file),'at+');
%         fprintf(txt_idx,"%05d,\t%s\n",i,strjoin(sample_names,', '));
%         fclose(txt_idx);

        audiowrite(fullfile(destination,filename),audio,Fs,BitsPerSample=16,title=label,Comment=all_labels);

    end

    multiWaitbar('fields', 'increment', paritions.(field));
end
multiWaitbar('fields','close');

end

