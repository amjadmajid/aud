clearvars

recordings_root = "..\Recorded_files\";
recordings_folder = uigetdir(recordings_root);

recordings = dir(recordings_folder + "/**/W_noise_32s/**/*.wav");

locations  = strings(size(recordings));
distances  = strings(size(recordings));
directions = strings(size(recordings));

for i = 1:length(recordings)
    metadata = strsplit(recordings(i).name(5:end-4),'_');
%     if strcmp(data(end-5:end-2), 'locA')
%         continue
%     end
        
    locations(i)  = metadata{end}(1:end-1);
    distances(i)  = metadata{1};
    directions(i) = metadata{2};
end

% removing empty strings
locations(cellfun('isempty',locations)) = [];
distances(cellfun('isempty',distances)) = [];
directions(cellfun('isempty',directions)) = [];

plot_hist(locations,recordings_folder)
plot_hist(distances,recordings_folder)
plot_hist(directions,recordings_folder)

function plot_hist(A,t)
[C,~,ic] = unique(A);
fig1 = figure;
axes1 = axes('Parent',fig1,'XTickLabel',C,'XTick',1:length(C));
hold(axes1,'on');
title(t,'Interpreter',"none")
histogram(ic)
end