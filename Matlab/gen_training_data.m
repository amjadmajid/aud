% generate training and validation data from the recorded samples,
% Data types:
%   - LoS: yes
%   - Sound type: white noise
%   - Sample length: 2s
% only line of sight with white noise for audio

clearvars
close all

rng(1) % for reproducability

train_val_ratio = 0.90;

Sample_path = "..\Audio_files\Audio_samples\Free_Top\Line_of_Sight\W_noise_32s\Samples_2s";
Samples = dir(sprintf("%s\\*.wav",Sample_path));

num_train_samples = floor(size(Samples,1)*train_val_ratio);
num_val_samples = size(Samples,1) - num_train_samples;
Samples_to_train_with = randperm(size(Samples,1),num_train_samples);

rec_info = audioinfo(sprintf("%s\\%s",Samples(1).folder,Samples(1).name));

Xtrain = zeros(rec_info.TotalSamples,6,1,num_train_samples);
Ytrain = zeros(num_train_samples,1);
Xvalidation = zeros(rec_info.TotalSamples,6,1,num_val_samples);
Yvalidation = zeros(num_val_samples,1);

itrain=1;
ivalidation=1;
for i = 1:size(Samples,1)
    [sound,~] = audioread(sprintf("%s\\%s",Samples(i).folder,Samples(i).name));
    DoA = str2double(Samples(i).name(11:13));   % direction of arival (degrees)
    dist = str2double(Samples(i).name(5:7));    % distance to source (cm)
    if any(i == Samples_to_train_with)
        Xtrain(:,:,1,itrain) = sound;
        Ytrain(itrain,1) = DoA;
        itrain = itrain + 1;
    else
        Xvalidation(:,:,1,ivalidation) = sound;
        Yvalidation(ivalidation,1) = DoA;
        ivalidation = ivalidation + 1;
    end
end

Yvalidation = categorical(Yvalidation);
Ytrain =  categorical(Ytrain);
save("trainingData", "Xtrain", "Ytrain","-v7.3")
save("validationData", "Xvalidation", "Yvalidation","-v7.3");
