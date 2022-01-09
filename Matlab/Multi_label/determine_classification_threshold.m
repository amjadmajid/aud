return
clearvars
load("trained_net_1-4_sources_4-1-22_validation_results.mat")

test_thresholds = linspace(0,1,201);

accuracies  = zeros(size(test_thresholds));
precisions  = zeros(size(test_thresholds));
recalls     = zeros(size(test_thresholds));
F1s         = zeros(size(test_thresholds));

for i = 1:length(test_thresholds)
    y_p = y_pred > test_thresholds(i);
    [accuracy, precision, recall, F1_score] = multilabel_eval(y_p,y_true);

    accuracies(i)   = double(gather(extractdata(accuracy)));
    precisions(i)   = double(gather(extractdata(precision)));
    recalls(i)      = double(gather(extractdata(recall)));
    F1s(i)          = double(gather(extractdata(F1_score)));

    multiWaitbar('test thresholds',test_thresholds(i));
  

end
multiWaitbar('closeall');
%%

hold off
%plot(test_thresholds,accuracies, DisplayName='Accuracy')
findpeaks(accuracies,test_thresholds,NPeaks=1)

hold on
%plot(test_thresholds,precisions, DisplayName='Precision')
findpeaks(precisions,test_thresholds,NPeaks=1)

%plot(test_thresholds,recalls, DisplayName='Recall')
findpeaks(recalls,test_thresholds,NPeaks=1)

%plot(test_thresholds,F1s,DisplayName='F1')
findpeaks(F1s,test_thresholds,NPeaks=1)

legend('Accuracy', '', 'Precision', '', 'Recall', 'F1', '',Location="northeast")

xlabel('Classification threshold')
ylim([0 1])
xlim([0 1])
