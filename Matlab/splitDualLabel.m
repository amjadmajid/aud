function [split, remainder] = splitDualLabel(audio_DS, label ,p)

labels = audio_DS.Labels.Properties.VariableNames;
if length(labels) ~= 2
    error("number of labels != 2")
end

if class(audio_DS.Labels{:,:}) ~= "categorical"
    error("Labels must be categorical")
end

label_idx = find(labels == label);
if isempty(label_idx)
    error("label not in datastore")
end

label_categories = categories(audio_DS.Labels.(labels{label_idx}));
num_categories = size(label_categories,1);

ads_for_split=cell(num_categories,1);
ads_for_rem=cell(num_categories,1);
split_size = 0;
rem_size = 0;

for i = 1:num_categories
    idx = audio_DS.Labels.(labels{label_idx})==label_categories{i};
    
    ads_sub = subset(audio_DS,idx);
    [spl, rem] = splitEachLabel(ads_sub,p,'randomized','TableVariable',labels{mod(label_idx,2)+1});
    
    ads_for_split{i} = spl;
    split_size = split_size + size(spl.Files,1);
    
    ads_for_rem{i} = rem;
    rem_size = rem_size + size(rem.Files,1);
end

split_src = cell(split_size,1);
split_labels = table('Size',[split_size,2],...
    'VariableTypes',{'categorical', 'categorical'},...
    'VariableNames',audio_DS.Labels.Properties.VariableNames);
split_i = 1;

rem_src = cell(rem_size,1);
rem_labels = table('Size',[rem_size,2],...
    'VariableTypes',{'categorical', 'categorical'},...
    'VariableNames',audio_DS.Labels.Properties.VariableNames);
rem_i = 1;

for i = 1:num_categories
    split_src(split_i: split_i+size(ads_for_split{i}.Files,1)-1) = ...
        ads_for_split{i}.Files;
    
    split_labels(split_i: split_i+height(ads_for_split{i}.Labels)-1,:) = ...
        ads_for_split{i}.Labels;
    split_labels.Properties.RowNames(split_i: split_i+height(ads_for_split{i}.Labels)-1) = ...
        ads_for_split{i}.Labels.Row;
    
    split_i = split_i + size(ads_for_split{i}.Files,1);
    
    
    rem_src(rem_i: rem_i + size(ads_for_rem{i}.Files,1)-1) = ...
        ads_for_rem{i}.Files;
    
    rem_labels(rem_i: rem_i+height(ads_for_rem{i}.Labels)-1,:) = ...
        ads_for_rem{i}.Labels;
    rem_labels.Properties.RowNames(rem_i: rem_i+height(ads_for_rem{i}.Labels)-1) = ...
        ads_for_rem{i}.Labels.Row;
    
    rem_i = rem_i + size(ads_for_rem{i}.Files,1);
end

split = audioDatastore(split_src, 'Labels',split_labels);
remainder = audioDatastore(rem_src, 'Labels',rem_labels);
end


