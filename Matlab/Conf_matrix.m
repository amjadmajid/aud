function [idx_predicted, idx_true] = Conf_matrix (net, DS, DS_type)

% derive labels
sample = read(DS);
reset(DS);
Labels = categories(sample{2}); 

% run prediction on data set
Ypredicted = predict_progress(net,DS);
[~, idx_predicted] = max(Ypredicted,[],2);
[~, idx_true] = max(onehotencode(DS.UnderlyingDatastores{1,1}.Labels,2), [], 2);


%% confusion matrix
C = confusionmat(idx_true, idx_predicted);
figure
cm = confusionchart(C,Labels(unique(idx_true)),'Title','Confusion matrix '+ DS_type + ' data');
sortClasses(cm,Labels(unique(idx_true)))
accuracy = sum(diag(C))/sum(C,'all');
fprintf("accuracy: %.2f%% \n", accuracy*100)
%% distance error
[edist, ~] = loc_errors(idx_predicted, idx_true, Labels);


% as cdf
figure
cdfplot(edist)
xlabel("distance error")
ylabel("cdf")
title("distance error cdf " + DS_type + " data")


% per truth label
[theta, rho, avg_dist_error] = error_per_label(idx_true, edist, Labels);
figure
polarscatter(deg2rad(theta), rho, 50, avg_dist_error, "filled");
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
colorbar
title("mean distance error per location " + DS_type + " data")

%% noise detection
if any(strcmp(Labels,"Noise"))
    noise_idx = find(strcmp(Labels,"Noise"));

    C_noise = confusionmat((idx_true==noise_idx),(idx_predicted == noise_idx));
    figure
    cn = confusionchart(C_noise,["signal","noise"],'Title','Confusion matrix noise differentiation '+ DS_type + ' data'); 
    sortClasses(cn,["signal", "noise"])
end

end

%% Helper function deriving error distance and angles from predictions
function [dists, angles] = loc_errors(idx_pred, idx_true, labels)
dists = zeros(size(idx_pred));
angles = zeros(size(idx_pred));

for i = 1:length(idx_pred)
    if idx_true(i) == idx_pred(i) || strcmp(labels(idx_pred(i)),"Noise") || strcmp(labels(idx_true(i)),"Noise")
        continue
    end
    [r_pred, theta_pred] = label2loc(labels{idx_pred(i)});
    [r_true, theta_true] = label2loc(labels{idx_true(i)});

    angles(i) = wrapTo180(theta_pred - theta_true);
    dists(i) = pdist_polar(r_true, theta_true, r_pred, theta_pred);
end

end

function [theta, rho, avg_dist_error] = error_per_label(idx_true, edist, labels)

num_classes = length(labels) - any(strcmp(labels,"Noise"));
Class_err = zeros(num_classes,1);
Class_count = zeros(num_classes,1);

for i = 1:length(idx_true)
    if strcmp(labels{idx_true(i)},"Noise")
        continue
    end

    Class_err(idx_true(i)) = Class_err(idx_true(i)) + edist(idx_true(i));
    Class_count(idx_true(i)) = Class_count(idx_true(i)) +1;

end

avg_dist_error = Class_err./Class_count;

theta = zeros(num_classes,1);
rho = zeros(num_classes,1);

for i = 1:num_classes
    [rho(i),theta(i)] = label2loc(labels{i});
end

not_testsed = isnan(avg_dist_error);
avg_dist_error(not_testsed) = [];
theta(not_testsed) = [];
rho(not_testsed) = [];

end

%% Helper function to derive the distance between two points in polar cooridinates
function dist = pdist_polar(r1, theta1, r2, theta2)
    dist = sqrt(r1^2 + r2^2 - 2 * r1 * r2 * cosd(theta1 - theta2));
end

%% Helper function converting label to angle and distance
function [r, theta] = label2loc(label)
    r = str2double(label(8:10));
    theta = str2double(label(1:3));
end

%% Helper function to do perdiction with progres estimation
% uses multiWaitbar from the add-on explorer
% https://nl.mathworks.com/matlabcentral/fileexchange/26589-multiwaitbar 

function Ypredicted = predict_progress(net, DS)
num_files = length(DS.UnderlyingDatastores{1}.Files);
batchSize = 500;
num_batches = ceil(num_files/batchSize);

Ypredicted = [];

multiWaitbar('Predicting','reset');
for i = 1: num_batches
    batch_DS = partition(DS,num_batches,i);
    Ybatch = predict(net,batch_DS,'MiniBatchSize', 100, 'ExecutionEnvironment', 'gpu');
    Ypredicted = [Ypredicted; Ybatch];

    multiWaitbar('Predicting','increment',batchSize/num_files);
end
multiWaitbar('CLOSEALL');

end
