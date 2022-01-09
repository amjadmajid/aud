function single_source_samp_gen(data,storage_root,start, audio_length, paritions, num_samples)

if ~isfolder(storage_root)
    mkdir(storage_root)
end

fields = fieldnames(data);

multiWaitbar('fields',0);
for j = 1:length(fields)
    field = fields{j};

    destination = fullfile(storage_root,field);
    if ~isfolder(destination)
        mkdir(destination)
    end

    multiWaitbar('single source', 0);
    for i = 1:length(data) %lengt(data) = num chrip types

        DS = getfield(data,{i},field);

        all_labels = categories(DS.Labels.location(1));
        all_labels(strcmp(all_labels, 'Noise')) = [];
        all_labels = strjoin(all_labels,'-');

        files = DS.Files(randperm(numel(DS.Files)));

        samples_required = ceil(num_samples * paritions.(field) / length(data));

        if numel(files) < samples_required
            error("not enough files")
        end

        parfor ii = 1:samples_required
            file = files{ii};

            % determine name

            % location data
            location = strsplit(file,'\');
            location = location{end}(1:end-4);
            location = erase(location, 'rec_');

            % chirp data
            chirp = strsplit(file,'\');
            chirp = chirp{end-2}(end);

            sample_name = strjoin({chirp,location},'_');


            num_chirps = num2str(1);

            filename = strcat('rec_',num_chirps);
            filename = strjoin({filename, sample_name},'[');

            filename = strcat(filename,'.wav');

            %determine if this file already exists, 
            if exist(fullfile(destination,filename),'file')
                continue
            end
            

            % determine label
            label = strsplit(location,'_');
            label = flip(label(1:2));

            if ~contains(label{1},'deg')
                label{1} = strcat(label{1},'deg');
            end


            label = strjoin(label,'_');

            if ~contains(all_labels, label)
                error("given label does not in all labels\n file: %s",file)
            end


            %read audio
            [audio, Fs] = audioread(file);
            %crop audio
            crop_range = floor(start*Fs) + (0 : ceil(audio_length*Fs));
            audio = audio(crop_range,:);

            % store audio, store label in the title field and a list of all
            % labels in the Comment field
            audiowrite(fullfile(destination,filename),audio,Fs,BitsPerSample=16,title=label,Comment=all_labels);


        end


        multiWaitbar('single source', 'increment', 1/length(data));
    end
    multiWaitbar('fields', 'increment', paritions.(field));

end
multiWaitbar('fields', 'close');
multiWaitbar('single source', 'close');
end