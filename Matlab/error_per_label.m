function [theta, rho, avg_error, max_err] = error_per_label(idx_true, edist, labels)

num_classes = length(labels) - any(strcmp(labels,"Noise"));
Class_err = zeros(num_classes,1);
Class_count = zeros(num_classes,1);
max_err = zeros(num_classes,1);

for i = 1:length(idx_true)
    if strcmp(labels{idx_true(i)},"Noise")
        continue
    end

    Class_err(idx_true(i)) = Class_err(idx_true(i)) + edist(i);
    Class_count(idx_true(i)) = Class_count(idx_true(i)) +1;

    max_err(idx_true(i)) = max(edist(i),max_err(idx_true(i)));

end

avg_error = Class_err./Class_count;

theta = zeros(num_classes,1);
rho = zeros(num_classes,1);

for i = 1:num_classes
    [rho(i),theta(i)] = label2loc(labels{i});
end

not_testsed = isnan(avg_error);
avg_error(not_testsed) = [];
theta(not_testsed) = [];
rho(not_testsed) = [];

end