%determine accuracy, precision, recal and F1 score for minibatch
% source: Gibaja, Eva, and Sebastián Ventura. "A tutorial on multilabel learning."
function [accuracy, precision, recall, F1_score] = multilabel_eval(Y_pred,Y_true)

numObservations = size(Y_true,2);

accuracy = dlarray(0);
precision = dlarray(0);
recall = dlarray(0);
F1_score = dlarray(0);

for i = 1:numObservations
    predicted_correct = sum(Y_true(:,i) & Y_pred(:,i));
    mean_pred_correct = predicted_correct/numObservations;

    if sum(Y_true(:,i) | Y_pred(:,i)) ~= 0
        accuracy = accuracy + mean_pred_correct/sum(Y_true(:,i) | Y_pred(:,i));
    end

    if sum(Y_true(:,i)) ~= 0
        recall = recall + mean_pred_correct/sum(Y_true(:,i));
    end

    if sum(Y_pred(:,i)) ~= 0
        precision = precision + mean_pred_correct/sum(Y_pred(:,i));
    end

    if sum(Y_true(:,i)) ~= 0 || sum(Y_pred(:,i)) ~= 0
        F1_score = F1_score + 2*mean_pred_correct/(sum(Y_true(:,i)) + sum(Y_pred(:,i)));
    end

end
end