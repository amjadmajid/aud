return
clearvars
load('trained_net_1-4_sources_4-1-22_test_results.mat')

%%
% Test_sets{1} ... Test_sets{4} are for 1 to 4 simultanious sources resp


% determine detection accuracies as #sources detected/#sources transmitted
% sources detected in steps: 0, 1, 2, 3, 4, 5, 6, >6

max_nr_sources = 4;
max_detections_bin = 10; % 0, 1, 2, 3, 4, 5, 6 and > 6

fractions = zeros(max_nr_sources, max_detections_bin+2);
C = linspecer(max_detections_bin+2);
for i = 1:4
    y_p = Test_sets{i}.Y_pred > 0.2;%Test_sets{i}.classification_th;

    num_samples = size(y_p,2);
    
    num_detected = sum(y_p,1);
    fractions(i,1:end-1) = sum(num_detected == [0:max_detections_bin]',2)/num_samples;
    fractions(i,end) = sum(num_detected > max_detections_bin,2)/num_samples;

end

labels = num2str((0:max_detections_bin)') + " sources detected";
labels(end+1) = sprintf(">%d sources detected", max_detections_bin);

b = bar(fractions*100,'stacked');
for i = 1:numel(b)
    b(i).FaceColor = C(i,:);
end

ylim([0, 100])
xlim([0.5,6])

xlabel("Sources transmitted")
ylabel("Fraction sources detected [%]")
legend(labels)

