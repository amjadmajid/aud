clearvars

%load datasets
load("H:\aud\Multi_label_samples\Testing_datastores.mat"')

%load network
load("trained_net_1-4_sources_4-1-22.mat","best_net");

%%
classification_threshold = 0.2;
miniBatchSize = 1024;

multiWaitbar('running tests',0);
for i = 1:length(Test_sets)

    multiWaitbar(char(Test_sets{i}.type),0);

    DS = Test_sets{i}.DS;

    Reduce = false;
    if Reduce
        fprintf("reducing DS size\n")
        %use only 1% of the data
        DS = shuffle(DS);
        DS = partition(DS,100,1);
    end    

    mbq = minibatchqueue(DS, ...
        'MiniBatchSize',miniBatchSize, ...
        'MiniBatchFcn', @minibatchfcn,...
        'MiniBatchFormat',{'SSCB', ''});

    num_batches = ceil(length(DS.Files)/miniBatchSize);

    %get predictions
    [Y_pred, Y_true] = run_net(best_net,mbq,labels,num_batches,char(Test_sets{i}.type));

    Y_pred = double(gather(extractdata(Y_pred)));
    Y_true = double(gather(extractdata(Y_true)));

    y_pred = Y_pred > classification_threshold;
    
    % get hop errors
    ehops = hop_error_multi_label(y_pred, Y_true, labels);

    % store results
    Test_sets{i}.Y_pred = Y_pred;
    Test_sets{i}.Y_true = Y_true;
    Test_sets{i}.classification_th = classification_threshold;

    Test_sets{i}.ehops = ehops;

    multiWaitbar('running tests', 'increment', 1/length(Test_sets));
end
multiWaitbar('closeall');


save('trained_net_1-4_sources_4-1-22_test_results.mat',"Test_sets")

%% Turn off the pc at if training ends at nigth time
c = fix(clock);
h = c(4);
if h >= 2 && h < 8
    system('shutdown -s')
end
%% Helper functions
function [Y_pred, Y_true] = run_net(net,mbq,labels,num_batches,waitbar_title)

multiWaitbar(char(waitbar_title), 0);
reset(mbq)

Y_pred = dlarray(zeros(length(labels),1),'CB');
Y_true = dlarray(zeros(length(labels),1),'CB');

while hasdata(mbq)
    [dlX, dlY_true] = next(mbq);

    if canUseGPU
        dlX = gpuArray(dlX);
    end

    batch_pred = predict(net,dlX);

    Y_pred = [Y_pred, batch_pred];
    Y_true = [Y_true, dlY_true];

    multiWaitbar(char(waitbar_title),'increment',1/num_batches);
end

Y_pred(:,1) = [];
Y_true(:,1) = [];


multiWaitbar(char(waitbar_title),'close');
end

function [X,Y] = minibatchfcn(XCell,YCell)

% Preprocess predictors.
X = cat(4,XCell{:});

% multi hot encode labels
Y = zeros(length(categories(YCell{1})), length(YCell));
for i = 1:length(YCell)
    [~, idx] = ismember(YCell{i},categories(YCell{i}));
    Y(idx,i) = 1;

end

end