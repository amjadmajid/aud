
clearvars
close all

paritions = struct("Train", 0.7,"Val", 0.15,"Test", 0.15);
Total_samples = 600000;
max_nr_sources = 4;
sapmles_per_nr_sources = Total_samples/max_nr_sources;
%% find and load datastores

sample_root = "../../Sampled_files/Obstructed_Top/";
sample_folders = "chirp_train_chirp_0s024_" + num2str([0:7]');

%line of sight samples
fprintf("collect datastores, LoS\n")
sample_folders_LoS = fullfile(sample_root,"Line_of_Sight",sample_folders,"Samples","datastores");

LoS_data = struct("Train", cell(8,1),"Val", cell(8,1),"Test", cell(8,1));
LoS_train_sum = 0;
LoS_val_sum = 0;
LoS_test_sum = 0;

parfor i = 1:8
    %find datastore files
    list = dir(fullfile(sample_folders_LoS(i),"DS*.mat"));

    % get most recent
    [~,index] = sortrows({list.date}.');
    list = list(index(end:-1:1));

    % extract data
    DS_file = fullfile(list(1).folder, list(1).name);

    d = load(DS_file,"data");

    LoS_data(i).Train = d.data.training_DS;
    LoS_data(i).Val = d.data.validation_DS;
    LoS_data(i).Test = d.data.testing_DS;

    LoS_train_sum = LoS_train_sum + length(d.data.training_DS.Files);
    LoS_val_sum = LoS_val_sum + length(d.data.validation_DS.Files);
    LoS_test_sum = LoS_test_sum + length(d.data.testing_DS.Files);

end

LoS_sum = struct("Train", LoS_train_sum,"Val", LoS_val_sum,"Test", LoS_test_sum);

%non line of sight samples
fprintf("collect datastores, NLoS\n")
sample_folders_NLoS = fullfile(sample_root,"Non_Line_of_Sight",sample_folders,"Samples","datastores");

NLoS_data = struct("Train", cell(8,1),"Val", cell(8,1),"Test", cell(8,1));
NLoS_train_sum = 0;
NLoS_val_sum = 0;
NLoS_test_sum = 0;

parfor i = 1:8
    %find datastore files
    list = dir(fullfile(sample_folders_NLoS(i),"DS*.mat"));

    % get most recent
    [~,index] = sortrows({list.date}.');
    list = list(index(end:-1:1));

    % extract data
    DS_file = fullfile(list(1).folder, list(1).name);

    d = load(DS_file,"data");

    NLoS_data(i).Train = d.data.training_DS;
    NLoS_data(i).Val = d.data.validation_DS;
    NLoS_data(i).Test = d.data.testing_DS;

    NLoS_train_sum = NLoS_train_sum + length(d.data.training_DS.Files);
    NLoS_val_sum = NLoS_val_sum + length(d.data.validation_DS.Files);
    NLoS_test_sum = NLoS_test_sum + length(d.data.testing_DS.Files);

end

NLoS_sum = struct("Train", NLoS_train_sum,"Val", NLoS_val_sum,"Test", NLoS_test_sum);

%% generate samples for multi source

Multi_source_sample_root = "../../Multi_label_samples";
base_start = 0.0248;
chirp_length = .024;

shift_range = [0,0.020];

% generate single source

Storage_root = fullfile(Multi_source_sample_root,"Single_source");


%Los
fprintf("gen single scource, los\n")
single_source_samp_gen(LoS_data, Storage_root, base_start, chirp_length, paritions, round(2/3*sapmles_per_nr_sources))


%NLoS
fprintf("gen single scource, nlos\n")
single_source_samp_gen(NLoS_data, Storage_root, base_start, chirp_length, paritions, round(1/3*sapmles_per_nr_sources))


%% generate multi source

multiWaitbar('multi source',1/max_nr_sources);
for i = 2:max_nr_sources
    destination = sprintf("%d_sources",i);
    destination = fullfile(Multi_source_sample_root,destination);

    if ~isfolder(destination)
        mkdir(destination)
    end

    multi_source_samp_gen(LoS_data, NLoS_data, LoS_sum, NLoS_sum, i, ...
        paritions, destination, sapmles_per_nr_sources, ...
        base_start, chirp_length, shift_range)
    multiWaitbar('multi source', 'increment',1/max_nr_sources);
end
multiWaitbar('closeall');

%%% Turn off the pc at if done at nigth time
% c = fix(clock);
% h = c(4);
% if h >= 2 && h < 8
%     system('shutdown -s')
% end

gen_datastore;
Multi_label_training
