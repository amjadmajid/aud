clearvars

root_folder = "C:\Users\caspe\Documents\TU_Delft\Master\Thesis\Matlab_ML\Audio_files\Multi_source_audio\";

folders = ["Single_source"; num2str((2:4)') + "_sources"];

dists = 50:50:250;
envs = ["FS", "OC", "IC"];

combinations = combvec(dists, 1:length(envs));
configurations = strings(size(combinations,2),1);
for i = 1:size(combinations,2)
    configurations(i) = sprintf("*%03dcm*%s*",combinations(1,i),envs(combinations(2,i)));
end




source_files = [fullfile(root_folder, folders, "test"); ...
    fullfile(root_folder,"Single_source","Test", configurations)];


types = [folders;configurations];


Test_sets = cell(1,length(source_files));

multiWaitbar('ds gen', 0);
for i = 1:length(source_files)
    fs = matlab.io.datastore.FileSet(source_files(i),'FileExtensions','.wav');
    Test_sets{i}.DS  = fileDatastore(fs,'FileExtensions','.wav','ReadFcn',@read_multi_labeled_audio);
    Test_sets{i}.type = types(i);
    multiWaitbar('ds gen', i/length(source_files));
end

fprintf("DS done\n")
data = read(Test_sets{1}.DS);
reset(Test_sets{1}.DS)

audio = data{1};
info = data{2};

labels = categories(info);

[sample_length, sample_channels] = size(audio);

fprintf("saving\n")
save(fullfile(root_folder, "Testing_datastores.mat"), "Test_sets", "labels", "sample_length", "sample_channels", '-v7.3')
fprintf("done\n")

multiWaitbar('closeall');

%% Helper functions
function data = read_multi_labeled_audio(filename)
    info = audioinfo(filename);

    Labelsstr = strsplit(info.Title,'-');
    total_labels = strsplit(info.Comment,'-');
    
    label = categorical(Labelsstr, total_labels);

    audio = audioread(filename,'double');

    %toroidal padding
    if true
        c_l = 2;
        c_r = 3;
        s_t = 0;
        s_b = 0;
        audio = toroidal_padding(audio,s_t,s_b,c_l,c_r);

        info.TotalSamples = info.TotalSamples + s_t + s_b;
        info.NumChannels = info.NumChannels + c_l + c_r;
    end
    
    data = {audio, label};
end

function Z = toroidal_padding(X, t, b, l, r)

inputSize = size(X);

persistent row_copy col_copy;
if isempty(row_copy)
    row_copy = loop_eye(inputSize(1), t, b);
end
if isempty(col_copy)
    col_copy = loop_eye(inputSize(2), l, r)';
end

Z = row_copy * X * col_copy;

end

function Y = loop_eye(base, t, b)

Y = zeros(base + [t+b,0]);
j = mod(-t, base);

for i = 1:size(Y,1)
    j = j +1;
    if j > base
        j = 1;
    end
    Y(i,j) = 1;
end

end
