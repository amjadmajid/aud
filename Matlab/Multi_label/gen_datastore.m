clearvars

folders = [num2str((2:4)') + "_sources";"Single_source"];
%base_folder = "H:\aud\Multi_label_samples\";
base_folder = "C:\Users\caspe\Documents\TU_Delft\Master\Thesis\Matlab_ML\Audio_files\Multi_source_audio\";

save_name = "1-4_sources_datastores";

reduce = false;

fs_train = matlab.io.datastore.FileSet(fullfile(base_folder, folders, "Train"),'FileExtensions','.wav');
fs_val = matlab.io.datastore.FileSet(fullfile(base_folder, folders, "Val"),'FileExtensions','.wav');
fs_test = matlab.io.datastore.FileSet(fullfile(base_folder, folders, "Test"),'FileExtensions','.wav');

if reduce
    max_total = 400000;
    total = fs_train.NumFiles + fs_val.NumFiles + fs_test.NumFiles;

    if max_total < total
        remainder_fraction = max_total/total;

        randIdx = randperm(fs_train.NumFiles);
        nIdxToKeep = round(remainder_fraction*fs_train.NumFiles);
        remainingIdx = randIdx(1:nIdxToKeep);
        fs_train = subset(fs_train,remainingIdx);

        randIdx = randperm(fs_val.NumFiles);
        nIdxToKeep = round(remainder_fraction*fs_val.NumFiles);
        remainingIdx = randIdx(1:nIdxToKeep);
        fs_val = subset(fs_val,remainingIdx);

        randIdx = randperm(fs_test.NumFiles);
        nIdxToKeep = round(remainder_fraction*fs_test.NumFiles);
        remainingIdx = randIdx(1:nIdxToKeep);
        fs_test = subset(fs_test,remainingIdx);


        new_total = fs_train.NumFiles + fs_val.NumFiles + fs_test.NumFiles;

    end
end


Train_ds = fileDatastore(fs_train,'FileExtensions','.wav','ReadFcn',@read_multi_labeled_audio);
Val_ds   = fileDatastore(fs_val,'FileExtensions','.wav','ReadFcn',@read_multi_labeled_audio);
Test_ds  = fileDatastore(fs_test,'FileExtensions','.wav','ReadFcn',@read_multi_labeled_audio);

fprintf("DS done\n")
data = read(Train_ds);
reset(Test_ds)

audio = data{1};
info = data{2};

labels = categories(info);

[sample_length, sample_channels] = size(audio);

fprintf("saving\n")

if exist(fullfile(base_folder,save_name)+".mat", 'file')
    i = 1;
    while exist(fullfile(base_folder, sprintf("%s_%d.mat",save_name,i)),'file')
        i = i +1;
    end
    save_name = sprintf("%s_%d",save_name,i);
end

save(fullfile(base_folder,save_name)+".mat", "Train_ds", "Val_ds", "Test_ds", "labels", "sample_length", "sample_channels", '-v7.3')
fprintf("done\n")

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
