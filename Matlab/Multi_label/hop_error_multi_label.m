function [ehops] = hop_error_multi_label(pred_set, truth_set, labels)
% uses VoronoiLimit: https://nl.mathworks.com/matlabcentral/fileexchange/34428-voronoilimit-varargin
% generate map
% with one aditional class at the origin

max_hop_error = 15;

loc_labels = labels(~strcmpi(labels, "Noise")); %remove noise label

locs = zeros(length(loc_labels)+1,2);
for i = 1:length(loc_labels)
    [locs(i,1), locs(i,2)] = label2loc(loc_labels{i});
end
locs(:,2) = deg2rad(locs(:,2));
[x, y] = pol2cart(locs(:,2),locs(:,1));


[v, c, ~] = VoronoiLimit(x,y,'figure','off');


ehops = zeros(size(pred_set,2),1);
for i = 1:size(pred_set,2)
    pred = pred_set(:,i);
    truth = truth_set(:,i);

    num_truth = sum(truth);
    num_pred = sum(pred);

    %fprintf("idx: %03d\t %d of %d\n", i,num_pred,num_truth);

    if num_pred == 0
        % no detection => hop error = max 
        ehops(i) = max_hop_error;
        continue
    end

    truth_locs = locs(logical(truth),:);
    pred_locs = locs(logical(pred),:);

    ehop = 0;

    % find the predictions closest to each truth label
    pairs = dsearchn(pred_locs,truth_locs);

    for j = 1: num_truth
        if all(pred_locs(pairs(j),:) == truth_locs(j,:))
            % correctly determined
            continue
        end

        [x_pred, y_pred] = pol2cart(deg2rad(pred_locs(pairs(j),2)), pred_locs(pairs(j),1));
        [x_true, y_true] = pol2cart(deg2rad(truth_locs(j,2)),       truth_locs(j,1));

        ehop = ehop + cells_passed(c, v, x_pred, y_pred, x_true, y_true,false);
    end

    non_pairs = find(~ismember(1:num_pred,pairs));

    for j = 1:length(non_pairs)
        % add ehop error to center for false positives

        [x_pred, y_pred] = pol2cart(deg2rad(pred_locs(non_pairs(j),2)), pred_locs(non_pairs(j),1));
        ehop = ehop + cells_passed(c, v, x_pred, y_pred, 0, 0,false);
    end

    ehops(i) = ehop/num_truth;
    
end

end