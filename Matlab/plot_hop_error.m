clearvars
close all

load('H:\aud\Matlab\Training_process_images\trained_net_location_13_12_2021\prediction_results_added_noise_5dB.mat')
idx_true = idx_true_test;
idx_pred = idx_pred_test;

load('label_definitions.mat')
labels = cellstr(labelDef(3).Categories);

DS_type = 'testdata + 5 SNR noise';

ehops = hop_error(idx_pred, idx_true, labels);

% as cdf
figure
cdfplot(ehops(ehops ~= -1))
xlabel("hop error")
ylabel("cdf")
title("hop error cdf " + DS_type + " data")


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
multiWaitbar('run', 0);
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

    ehops(i) = cells_passed(c, v, x_pred, y_pred, x_true, y_true,mod(i,100) == 0);

    multiWaitbar('run', 'increment', 1/length(idx_pred));
end

end

%% Helper function converting label to angle and distance
    function [r, theta] = label2loc(label)
        r = str2double(label(8:10));
        theta = str2double(label(1:3));
    end
