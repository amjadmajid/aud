function [idx_predicted, idx_true] = Conf_matrix (net, DS, DS_type)

% derive labels
sample = read(DS);
reset(DS);
Labels = categories(sample{2});

side_load = false

if ~side_load
    %run prediction on data set
    Ypredicted = predict_progress(net,DS,char(DS_type));
    [~, idx_predicted] = max(Ypredicted,[],2);
    [~, idx_true] = max(onehotencode(DS.UnderlyingDatastores{1,1}.Labels,2), [], 2);
else
    load('H:\aud\Matlab\Training_process_images\trained_net_location_13_12_2021\prediction_results.mat')
    idx_predicted = idx_pred_train;
    idx_true = idx_true_train;
end
Used_labels = Labels(unique(idx_true));
%% confusion matrix
C = confusionmat(idx_true, idx_predicted);
figure
cm = confusionchart(C,Used_labels,'Title','Confusion matrix '+ DS_type + ' data');
sortClasses(cm,Used_labels)
accuracy = sum(diag(C))/sum(C,'all');
fprintf("accuracy: %.2f%% \n", accuracy*100)
%% distance error
[edist, ~] = loc_errors(idx_predicted, idx_true, Used_labels);

% as cdf
figure
cdfplot(edist(edist ~= -1))
xlabel("distance error")
ylabel("cdf")
title("distance error cdf " + DS_type + " data")


% per truth label
[theta, rho, avg_dist_error, max_dist_error] = error_per_label(idx_true, edist, Used_labels);
figure
polarscatter(deg2rad(theta), rho, 50, avg_dist_error, "filled");
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
colorbar
title("mean distance error per location " + DS_type + " data")

figure
polarscatter(deg2rad(theta), rho, 50, max_dist_error, "filled");
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
colorbar
title("maximum distance error per location " + DS_type + " data")

%% noise detection
if any(strcmp(Used_labels,"Noise"))
    noise_idx = find(strcmp(Used_labels,"Noise"));

    C_noise = confusionmat((idx_true==noise_idx),(idx_predicted == noise_idx));
    figure
    cn = confusionchart(C_noise,["signal","noise"],'Title','Confusion matrix noise differentiation '+ DS_type + ' data');
    sortClasses(cn,["signal", "noise"])
end

%% Hops error
ehops = hop_error(idx_predicted, idx_true, Used_labels);
% as cdf
figure
cdfplot(ehops(ehops ~= -1))
xlabel("hops error")
ylabel("cdf")
title("hops error cdf " + DS_type + " data")

% per truth label
[theta, rho, avg_hop_error, max_hop_error] = error_per_label(idx_true, ehops, Used_labels);
figure
polarscatter(deg2rad(theta), rho, 50, avg_hop_error, "filled");
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
colorbar
title("mean hop error per location " + DS_type + " data")

figure
polarscatter(deg2rad(theta), rho, 50, max_hop_error, "filled");
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
colorbar
title("maximum hop error per location " + DS_type + " data")
end

%% Helper function deriving error distance and angles from predictions
function [dists, angles] = loc_errors(idx_pred, idx_true, labels)
dists = zeros(size(idx_pred));
angles = zeros(size(idx_pred));

for i = 1:length(idx_pred)
    if idx_true(i) == idx_pred(i)
        continue
    elseif strcmp(labels(idx_pred(i)),"Noise") || strcmp(labels(idx_true(i)),"Noise")
        angles(i) = -1;
        dists(i) = -1;
        continue
    end
    [r_pred, theta_pred] = label2loc(labels{idx_pred(i)});
    [r_true, theta_true] = label2loc(labels{idx_true(i)});

    angles(i) = wrapTo180(theta_pred - theta_true);
    dists(i) = pdist_polar(r_true, theta_true, r_pred, theta_pred);
end

end


%% Helper function to derive the distance between two points in polar cooridinates
    function dist = pdist_polar(r1, theta1, r2, theta2)
        dist = sqrt(r1^2 + r2^2 - 2 * r1 * r2 * cosd(theta1 - theta2));
    end


%% Helper function to do perdiction with progres estimation
% uses multiWaitbar from the add-on explorer
% https://nl.mathworks.com/matlabcentral/fileexchange/26589-multiwaitbar

    function Ypredicted = predict_progress(net, DS, opt_title)

        if nargin > 2
            title = opt_title;
        else
            title = "";
        end

        num_files = length(DS.UnderlyingDatastores{1}.Files);
        batchSize = 500;
        num_batches = ceil(num_files/batchSize);

        Ypredicted = [];

        multiWaitbar(append('Predicting: ',title),'reset');
        for i = 1: num_batches
            batch_DS = partition(DS,num_batches,i);
            Ybatch = predict(net,batch_DS,'MiniBatchSize', 100, 'ExecutionEnvironment', 'gpu');
            Ypredicted = [Ypredicted; Ybatch];

            multiWaitbar(append('Predicting: ',title),'increment',batchSize/num_files);
        end
        multiWaitbar('CLOSEALL');

    end
