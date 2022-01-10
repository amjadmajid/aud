clearvars
close all
%% Load audio data
%load H:\aud\Multi_label_samples\1-4_sources_datastores_2.mat
%load H:\aud\Multi_label_samples\full_datastores.mat
load C:\Users\caspe\Documents\TU_Delft\Master\Thesis\Matlab_ML\Audio_files\Multi_source_audio\1-4_sources_datastores_reduced.mat

Reduce = false;
if Reduce
    %use only 1% of the data
    Train_ds = shuffle(Train_ds);
    Train_ds = partition(Train_ds,100,1);

    Val_ds = shuffle(Val_ds);
    Val_ds = partition(Val_ds,100,1);

end



%% define network

Input_layer_size = [sample_length, sample_channels ,1];
Output_layer_size = length(labels);

% treshold for determining whether to give the class a positive response
labelThreshold = 0.5;

% side_load network for continuation of training or layer input/
side_load = false;
if side_load
    load trained_net_1-4_sources_09-1-22.mat

    % add two fully connected layers before the final layer
    lgraph = layerGraph(best_net);

    lnew = [fullyConnectedLayer(64,'name', 'fc_new_1','WeightLearnRateFactor',50, 'BiasLearnRateFactor',50),...
        reluLayer('name', 'relu_new_1'),...
        fullyConnectedLayer(64,'name', 'fc_new_2','WeightLearnRateFactor',50, 'BiasLearnRateFactor',50),...
        reluLayer('name', 'relu_new_2'),...
        fullyConnectedLayer(Output_layer_size,'name', 'fc_new_out','WeightLearnRateFactor',50, 'BiasLearnRateFactor',50)];

    lgraph_new = replaceLayer(lgraph,'fc_2', lnew);


    dlnet = dlnetwork(lgraph_new);

    clear best_net

else

    layers = [
        imageInputLayer(Input_layer_size,"Name","Input","Normalization","none")

        %dropoutLayer(0.4)
        convolution2dLayer([20 6],32)
        reluLayer()

        convolution2dLayer([50 3],32)
        reluLayer()

        convolution2dLayer([50 3],32)
        reluLayer()

        convolution2dLayer([50 2],32)
        reluLayer()

        fullyConnectedLayer(128)
        reluLayer()

        fullyConnectedLayer(Output_layer_size)
        sigmoidLayer()
        ];

    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);

end
%% Training setup

% training options
miniBatchSize = 1024;
numEpochs   = 20;
gradDecay   = 0.9;
sqGradDecay = 0.999;
epsilon     = 1.0e-08;
l2Regularization = 0.005;

if side_load
    % low learning rate for transfer learning (the settign of the weight
    % factors in the new layers gives those a higher learning rate whilst
    % the old layers are smaller
    learnRate = 1e-4;
else
    learnRate = 0.005;
end
    

batch_per_epoch = ceil(length(Train_ds.Files)/miniBatchSize);
% have two validation steps per epoch
validation_frequency = floor(batch_per_epoch/2);

% validation_patience specifies the number of times that the loss on the
% validation set can be larger than or equal to the previously smallest
% loss before network training stops.
validation_patience = 5;

plots = "training-progress";

executionEnvironment = "auto";

train_mbq = minibatchqueue(Train_ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFcn', @minibatchfcn,...
    'MiniBatchFormat',{'SSCB', ''});

if validation_frequency > 0
    val_mbq = minibatchqueue(Val_ds, ...
        'MiniBatchSize',miniBatchSize, ...
        'MiniBatchFcn', @minibatchfcn,...
        'MiniBatchFormat',{'SSCB', ''});

    batch_per_val = ceil(length(Val_ds.Files)/miniBatchSize);
end

if plots == "training-progress"
    figure('Position',[37,74,1396,882])

    % Labeling multi label evaluation.
    subplot(2,1,1)
    line_acc_train =        animatedline('Color', '#0072BD','DisplayName','Accuracy');
    line_precision_train =  animatedline('Color', '#D95319','DisplayName','Precision');
    line_recall_train =     animatedline('Color', '#EDB120','DisplayName','Recall');
    line_F1_train =         animatedline('Color', '#7E2F8E','DisplayName','F1 score');

    if validation_frequency > 0
        line_acc_val =          animatedline('Color', '#0072BD', ...
            'LineStyle','--', 'Marker','o', 'MarkerFaceColor', 'auto');
        line_precision_val =    animatedline('Color', '#D95319', ...
            'LineStyle','--', 'Marker','o', 'MarkerFaceColor', 'auto');
        line_recall_val =       animatedline('Color', '#EDB120', ...
            'LineStyle','--', 'Marker','o', 'MarkerFaceColor', 'auto');
        line_F1_val =           animatedline('Color', '#7E2F8E', ...
            'LineStyle','--', 'Marker','o', 'MarkerFaceColor', 'auto');
    end

    legend('Location','southeast')
    ylim([0 1])
    xlim([0 numEpochs])
    xlabel("Epoch")
    grid on

    % Loss.
    subplot(2,1,2)

    line_loss_train = animatedline('Color','#0072BD');

    if validation_frequency > 0
        line_loss_val = animatedline( 'Color','#0072BD', ...
            'LineStyle','--', 'Marker','o', 'MarkerFaceColor','auto');
    end
    ylim([0 inf])
    xlim([0 numEpochs])
    xlabel("Epoch")
    ylabel("Loss")
    grid on
end

if (executionEnvironment == "auto" || executionEnvironment == "gpu") && canUseGPU
    fprintf("training on GPU\n");
else
    fprintf("training on CPU\n");
end

%% Training

best_valdiataion_loss = inf;
val_patencience_reached = false;
Val_done = false;
val_patience_count = 0;

averageGrad = [];
averageSqGrad = [];

iteration = 0;
start = tic;

multiWaitbar('Training',0, 'CanCancel', 'on');

for epoch = 1:numEpochs
    %shuffle data
    shuffle(train_mbq)

    while hasdata(train_mbq)
        iteration = iteration + 1;

        % get minibatch
        [dlX, dlY_true] = next(train_mbq);

        % If training on a GPU, then convert data to a gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function.
        [grad,loss,dlY_pred] = dlfeval(@modelGradients,dlnet,dlX,dlY_true,l2Regularization);

        % Run validation
        if validation_frequency > 0 && (iteration == 1 || mod(iteration,validation_frequency) == 0)

            [Y_pred, Y_true] = run_validation(dlnet,val_mbq,labels,batch_per_val);

            multiWaitbar('computing validation metrics','Busy');

            % Loss.
            val_loss = crossentropy(Y_pred,Y_true,'TargetCategories','independent');
            fprintf("validation loss: %.4f\t\t",val_loss);

            % Multi label metrics
            Y_pred = extractdata(Y_pred) > labelThreshold;
            [accuracy, precision, recall, F1_score] = multilabel_eval(Y_pred,Y_true);

            % test if validation loss improved
            if val_loss < best_valdiataion_loss
                fprintf("network improvement\n");

                best_valdiataion_loss = val_loss;
                best_net = dlnet;
                best_net_iteration = iteration;

                val_patience_count = 0;
            elseif val_patience_count > validation_patience
                val_patencience_reached = true;
            else
                val_patience_count = val_patience_count + 1;
                fprintf("no improvement, patience count: %d\n",val_patience_count);

            end


            multiWaitbar('computing validation metrics','close');

            Val_done = true;
        end


        % update network using adam optimizer
        [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grad, ...
            averageGrad,averageSqGrad,iteration);%,learnRate,gradDecay,sqGradDecay);



        % plotting
        if plots == "training-progress"
            subplot(2,1,1)
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title("Epoch: " + epoch + ", Elapsed: " + string(D))

            % Loss.
            addpoints(line_loss_train,iteration/batch_per_epoch,double(gather(extractdata(loss))))

            % evaluation scores.
            Y_pred = extractdata(dlY_pred) > labelThreshold;
            [accuracy, precision, recall, F1_score] = multilabel_eval(Y_pred,dlY_true);

            addpoints(line_acc_train,       iteration/batch_per_epoch, double(gather(extractdata(accuracy))))
            addpoints(line_precision_train, iteration/batch_per_epoch, double(gather(extractdata(precision))))
            addpoints(line_recall_train,    iteration/batch_per_epoch, double(gather(extractdata(recall))))
            addpoints(line_F1_train,        iteration/batch_per_epoch, double(gather(extractdata(F1_score))))

            if Val_done
                Val_done = false;

                addpoints(line_loss_val,iteration/batch_per_epoch,double(gather(extractdata(val_loss))))

                addpoints(line_acc_val,       iteration/batch_per_epoch, double(gather(extractdata(accuracy))))
                addpoints(line_precision_val, iteration/batch_per_epoch, double(gather(extractdata(precision))))
                addpoints(line_recall_val,    iteration/batch_per_epoch, double(gather(extractdata(recall))))
                addpoints(line_F1_val,        iteration/batch_per_epoch, double(gather(extractdata(F1_score))))

            end
            drawnow


        end

        abort = multiWaitbar( 'Training', 'increment', 1/(numEpochs*batch_per_epoch) );
        if abort || val_patencience_reached
            break
        end
    end

    if abort|| val_patencience_reached
        multiWaitbar('Training', 'Relabel', 'Training aborted');

        % final validation step

        if validation_frequency > 0
            multiWaitbar('aborted validation','Busy');

            [Y_pred, Y_true] = run_validation(dlnet,val_mbq,labels,batch_per_val);

            % Loss.
            val_loss = crossentropy(Y_pred,Y_true,'TargetCategories','independent');
            fprintf("validation loss: %.4f\t",val_loss);

            % Multi label metrics
            Y_pred = extractdata(Y_pred) > labelThreshold;
            [accuracy, precision, recall, F1_score] = multilabel_eval(Y_pred,Y_true);

            if plots == "training-progress"

                addpoints(line_loss_val,iteration/batch_per_epoch,double(gather(extractdata(val_loss))))

                addpoints(line_acc_val,       iteration/batch_per_epoch, double(gather(extractdata(accuracy))))
                addpoints(line_precision_val, iteration/batch_per_epoch, double(gather(extractdata(precision))))
                addpoints(line_recall_val,    iteration/batch_per_epoch, double(gather(extractdata(recall))))
                addpoints(line_F1_val,        iteration/batch_per_epoch, double(gather(extractdata(F1_score))))

                drawnow
            end

            if val_loss < best_valdiataion_loss
                fprintf("network improvement");

                best_valdiataion_loss = val_loss;
                best_net = dlnet;
                best_net_iteration = iteration;
            end
            fprintf("\n");
            multiWaitbar('aborted validation','close');
        end

        break
    end
end

if validation_frequency == 0
    % store last net
    best_net = dlnet;
end

fprintf("training complete\n");

date = string(datetime('today','Format','dd_MM_yyyy'));
save_name = "trained_net_1-4_sources_" + date;

save(save_name +".mat","best_net")

if validation_frequency > 0
    % compute results for best net

    [Y_pred, Y_true] = run_validation(best_net,val_mbq,labels,batch_per_val);

    val_loss = crossentropy(Y_pred,Y_true,'TargetCategories','independent');

    % Multi label metrics
    Y_pred = extractdata(Y_pred) > labelThreshold;
    [accuracy, precision, recall, F1_score] = multilabel_eval(Y_pred,Y_true);

    fprintf("Best net scores:\n" + ...
        "\tvalidation loss: %1.3f\n" + ...
        "\taccuracy: %1.3f\n" + ...
        "\tprecision: %1.3f\n" + ...
        "\trecall: %1.3f\n" + ...
        "\tF1_score: %1.3f\n", ...
        val_loss,accuracy, precision, recall, F1_score);

    if plots == "training-progress"
        subplot(2,1,1)
        hold on
        scatter(best_net_iteration/batch_per_epoch*ones(4),double(gather(extractdata([accuracy, precision, recall, F1_score]))),'filled','k');

        subplot(2,1,2)
        hold on
        scatter(best_net_iteration/batch_per_epoch, double(gather(extractdata(val_loss))),'filled','k')

        if ~isfolder("Training_images")
            mkdir("Training_images")
        end
        savefig(fullfile("Training_images",save_name))
    end
end
multiWaitbar('closeall');


%% Turn off the pc at if training ends at nigth time
c = fix(clock);
h = c(4);
if h >= 2 && h < 8
    system('shutdown -s')
end


%% Helper functions
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

function [gradients,loss,dlYPred] = modelGradients(dlnet,dlX,Y,l2Regularization)

dlYPred = forward(dlnet,dlX);

% crossentropy loss to determine performence
loss = crossentropy(dlYPred,Y,'TargetCategories','independent');

gradients = dlgradient(loss,dlnet.Learnables);

idx = dlnet.Learnables.Parameter == "Weights";
gradients(idx,:) = dlupdate(@(g,w) g + l2Regularization*w, gradients(idx,:), dlnet.Learnables(idx,:));

end

function [Y_pred, Y_true] = run_validation(dlnet,val_mbq,labels,batch_per_val)

multiWaitbar('validating', 0);
reset(val_mbq)

Y_pred = dlarray(zeros(length(labels),1),'CB');
Y_true = dlarray(zeros(length(labels),1),'CB');
while hasdata(val_mbq)
    [dlX, dlY_true] = next(val_mbq);

    if canUseGPU
        dlX = gpuArray(dlX);
    end

    batch_pred = predict(dlnet,dlX);

    Y_pred = [Y_pred, batch_pred];
    Y_true = [Y_true, dlY_true];

    multiWaitbar('validating','increment',1/batch_per_val);
end

Y_pred(:,1) = [];
Y_true(:,1) = [];

multiWaitbar('validating','close');
end