%clearvars
load('H:\aud\Matlab\Trained_nets\trained_net_location_13_12_2021.mat', 'Test_DS', 'net')
audio_DS = Test_DS.UnderlyingDatastores{1};
clear Test_DS;

Labels = categories(audio_DS.Labels);

noise_floors_to_test = [-90,-80, -70, -60, -50];
Waitbar_titles = char("Progress: " + num2str(noise_floors_to_test') + "dB");

for i = 1:length(noise_floors_to_test)
    multiWaitbar(Waitbar_titles(i,:),'reset');
end


results = cell(size(noise_floors_to_test));

for i = 1:length(noise_floors_to_test)
    %generate DS
    n_dB = noise_floors_to_test(i);
    DS_n = transform(audio_DS, @(data, info) preprocessForTraining_noise(data, info, n_dB), 'IncludeInfo', true);

    Ypredicted = predict_progress(net, DS_n, Waitbar_titles(i,:));

    [~, idx_predicted] = max(Ypredicted,[],2);
    [~, idx_true] = max(onehotencode(DS_n.UnderlyingDatastores{1,1}.Labels,2), [], 2);

    clear DS_n

    C = confusionmat(idx_true, idx_predicted);
    accuracy = sum(diag(C))/sum(C,'all');
    fprintf("Noise floor: %ddB - accuracy: %.2f%% \n",noise_floors_to_test(i), accuracy*100)

    ehops = hop_error(idx_predicted, idx_true, Labels(unique(idx_true)));

    % as cdf
    figure
    cdfplot(ehops(ehops ~= -1))
    xlabel("hop error")
    ylabel("cdf")
    title(sprintf("hop error cdf, noise floor: %ddb",noise_floors_to_test(i)))

    results{i}.noise_floor = noise_floors_to_test(i);
    results{i}.idx_predicted = idx_predicted;
    results{i}.idx_true = idx_true;
    results{i}.hop_errors = ehops;
end

%save("Training_process_images\trained_net_location_13_12_2021\noised_predictions_results","results")

multiWaitbar('closeall');

function [dataOut,info] = preprocessForTraining_noise(data,info, n_dB)

%add noise
[pxx,f] = periodogram(data(:,1),[],[],44100);
noise_band = 10*log10(bandpower(pxx,f,[11,20]*1000,'psd')/9000);

if noise_band < n_dB
    data = data + wgn(size(data,1), size(data,2), n_dB +43);
else
    disp skip
end

%loop around
if true
    data = toroidal_padding(data, 0,0,2,3);
end

dataOut = {data',info.Label};

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

%% Helper function to do perdiction with progres estimation
% uses multiWaitbar from the add-on explorer
% https://nl.mathworks.com/matlabcentral/fileexchange/26589-multiwaitbar

function Ypredicted = predict_progress(net, DS, bar_title)

if nargin > 2
    title = char(bar_title);
else
    title = 'predicting';
end

num_files = length(DS.UnderlyingDatastores{1}.Files);
batchSize = 500;
num_batches = ceil(num_files/batchSize);

Ypredicted = [];

multiWaitbar(title,'reset');
for i = 1: num_batches
    batch_DS = partition(DS,num_batches,i);
    Ybatch = predict(net,batch_DS,'MiniBatchSize', 100, 'ExecutionEnvironment', 'gpu');
    Ypredicted = [Ypredicted; Ybatch];

    multiWaitbar(title,'increment',batchSize/num_files);
end
multiWaitbar(title,'Close');

end

function ehops = hop_error(idx_pred, idx_true, labels)
% uses VoronoiLimit: https://nl.mathworks.com/matlabcentral/fileexchange/34428-voronoilimit-varargin
% generate map
% with one aditional class at the origin

loc_labels = labels(~strcmpi(labels, "Noise")); %remove noise label

locs = zeros(length(loc_labels)+1,2);
for i = 1:length(loc_labels)
    [locs(i,1), locs(i,2)] = label2loc(loc_labels{i});
end
locs(:,2) = deg2rad(locs(:,2));
[x, y] = pol2cart(locs(:,2),locs(:,1));


[v, c, ~] = VoronoiLimit(x,y,'figure','off');


ehops = zeros(size(idx_pred));
for i = 1:length(idx_pred)
    if idx_true(i) == idx_pred(i)
        continue
    elseif strcmp(labels(idx_pred(i)),"Noise") || strcmp(labels(idx_true(i)),"Noise")
        ehops(i) = -1;
        continue
    end
    [r_pred, theta_pred] = label2loc(labels{idx_pred(i)});
    [x_pred, y_pred] = pol2cart(deg2rad(theta_pred), r_pred);

    [r_true, theta_true] = label2loc(labels{idx_true(i)});
    [x_true, y_true] = pol2cart(deg2rad(theta_true), r_true);

    ehops(i) = cells_passed(c, v, x_pred, y_pred, x_true, y_true,false);
end

end

%% Helper function converting label to angle and distance
function [r, theta] = label2loc(label)
r = str2double(label(8:10));
theta = str2double(label(1:3));
end