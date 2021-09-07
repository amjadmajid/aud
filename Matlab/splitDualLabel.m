function [split, remainder] = splitDualLabel(adsi, labelDef, dir,p)

if ~isa(labelDef,'signalLabelDefinition')
    error("labelDef is not a signalLabelDefinition")
end
if size(labelDef,1) ~= 2
    error("number of labels != 2")
end
for i = 1:2
    if labelDef(i).LabelDataType ~= "categorical"
        error("Label '%s' is not categorical", labelDef(i).Name)
    end
end

if dir == 1
    label1 = labelDef(1);
    label2 = labelDef(2);
else
    label1 = labelDef(2);
    label2 = labelDef(1);
end

ads_for_split=cell(size(label1.Categories,1),1);
ads_for_rem=cell(size(label1.Categories,1),1);
split_size = 0;
rem_size = 0;

for i = 1:size(label1.Categories,1)
    idx = adsi.Labels.(label1.Name)==label1.Categories(i);
    
    ads_sub = subset(adsi,idx);
    [spl, rem] = splitEachLabel(ads_sub,p,'randomized','TableVariable',label2.Name);
    
    ads_for_split{i} = spl;
    split_size = split_size + size(spl.Files,1);
    
    ads_for_rem{i} = rem;
    rem_size = rem_size + size(rem.Files,1);
end

split_src = cell(split_size,1);
split_labels = table('Size',[split_size,2],...
    'VariableTypes',{'categorical', 'categorical'},...
    'VariableNames',adsi.Labels.Properties.VariableNames);
split_i = 1;

rem_src = cell(rem_size,1);
rem_labels = table('Size',[rem_size,2],...
    'VariableTypes',{'categorical', 'categorical'},...
    'VariableNames',adsi.Labels.Properties.VariableNames);
rem_i = 1;

for i = 1:size(label1.Categories,1)
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


