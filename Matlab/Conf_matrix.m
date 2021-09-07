clearvars
close all

sound = "white";
label = "distance";

if sound == "white"
    Sample_path = "..\Sampled_files\Free_Top\Line_of_Sight\W_noise_32s\Samples_2s\";
elseif sound == "chirp"
    Sample_path = "..\Sampled_files\Free_Top\Line_of_Sight\chirps_20_to_20k_in_s314__32s\Samples_2s\";
else
    error("unknown sound")
end

load("label_definitions.mat")

for i = 1:size(labelDef,1)
    notfound = false;
    if labelDef(i,1).Name == label
        Labels = labelDef(i,1).Categories; 
        break
    else 
        notfound = true;
    end
end

if notfound
    error("unknown label")
end

load(Sample_path+"training_"+ label + "_DB.mat")
load(Sample_path+"validation_"+ label + "_DB.mat")

load("trained_net_chirp_direction_31_08_2021.mat")

%% Test on validation data

Ypredicted = predict(net,validation_DB,'MiniBatchSize', 36, 'ExecutionEnvironment', 'gpu');
[~, idx_predicted] = max(Ypredicted,[],2);
[~, idx_true] = max(onehotencode(validation_DB.UnderlyingDatastores{1,1}.Labels,2), [], 2);
C_validation = confusionmat(idx_true, idx_predicted);

figure
cm_val = confusionchart(C_validation,Labels,'Title','Confusion matrix validation data');
sortClasses(cm_val,Labels)
acc_val = sum(diag(C_validation))/sum(C_validation,'all')

%% test on training data

Ypredicted = predict(net,training_DB,'MiniBatchSize', 36, 'ExecutionEnvironment', 'gpu');
[~, idx_predicted] = max(Ypredicted,[],2);
[~, idx_true] = max(onehotencode(training_DB.UnderlyingDatastores{1,1}.Labels,2), [], 2);
C_training = confusionmat(idx_true, idx_predicted);

figure
cm_train = confusionchart(C_training,Labels,'Title','Confusion matrix training data');
sortClasses(cm_train,Labels)
acc_train = sum(diag(C_training))/sum(C_training,'all')
