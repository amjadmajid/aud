clearvars
close all

load label_definitions.mat

Training_label = "direction";

%% load network

[file, path]= uigetfile("Trained_nets/*.mat","select nn");

vars = {'Test_DS'; 'Train_DS'; 'Val_DS';'net';'sample_metadata'; 'training_options'};

if file == 0
    return
elseif ~all(strcmp(who('-file',fullfile(path, file)),vars))
    
    fprintf("invalid file structure\n")
    return
else
    vars = {'Train_DS'; 'Val_DS';'net';'sample_metadata'};
    load(fullfile(path,file),vars{:})
    fprintf("%s\n", fullfile(path, file))
end

sample = read(Train_DS);
reset(Train_DS);
Labels = categories(sample{2}); 

%% Test on validation data

Ypredicted = predict(net,Val_DS,'MiniBatchSize', 64, 'ExecutionEnvironment', 'gpu');
[~, idx_predicted] = max(Ypredicted,[],2);
[~, idx_true] = max(onehotencode(Val_DS.UnderlyingDatastores{1,1}.Labels,2), [], 2);
C_validation = confusionmat(idx_true, idx_predicted);

figure
cm_val = confusionchart(C_validation,Labels,'Title','Confusion matrix validation data');
sortClasses(cm_val,Labels)
acc_val = sum(diag(C_validation))/sum(C_validation,'all')

%% test on training data

Ypredicted = predict(net,Train_DS,'MiniBatchSize', 36, 'ExecutionEnvironment', 'gpu');
[~, idx_predicted] = max(Ypredicted,[],2);
[~, idx_true] = max(onehotencode(Train_DS.UnderlyingDatastores{1,1}.Labels,2), [], 2);
C_training = confusionmat(idx_true, idx_predicted);

figure
cm_train = confusionchart(C_training,Labels,'Title','Confusion matrix training data');
sortClasses(cm_train,Labels)
acc_train = sum(diag(C_training))/sum(C_training,'all')
