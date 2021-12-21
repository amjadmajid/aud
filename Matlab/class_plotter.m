clearvars
close all

locs = zeros(96, 2);

for i = 1:length(locs)
    if i <= 24
        locs(i,1) = mod(i-1,12)*30;
        locs(i,2) = ceil(i/12)*0.50;
    else
        
        locs(i,1) = mod(i-1,24)*15;
        locs(i,2) = (1+ceil(i/24))*0.50;
    end
end

% locs = zeros(60, 2);
% 
% for i = 1:length(locs)
% 
%         locs(i,1) = mod(i-1,12)*30;
%         locs(i,2) = ceil(i/12)*0.20;
% 
% end

locs(:,1) = deg2rad(locs(:,1));

%Scatter plot
polarscatter(locs(:,1),locs(:,2),'filled');
hold on

%voronoi
[x, y] = pol2cart(locs(:,1),locs(:,2));
[v, c, xy] = VoronoiLimit(x,y,'bs_ext', 3.1*[1 1 -1 -1 1; 1 -1 -1 1 1]','figure','off');
hold on
A = zeros(length(c),1);
for i = 1:length(c)
    v1 = v([c{i}(end);c{i}],1); 
    v2 = v([c{i}(end);c{i}],2);

    [t,r] = cart2pol(v1',v2');

    polarplot(t,r,'Color',[0 0.4470 0.7410]+0.2)

    A(i) = polyarea(v1,v2) ;
end

%Center
polarscatter(0,0,500,[0.4660 0.6740 0.1880],'filled');

r = 0.15*ones(1,8);
D_theta = (2*pi)/6;
theta = 0:D_theta:(2*pi+ D_theta);
theta = theta - D_theta/2;
polarplot(theta,r,Color=[0.4660 0.6740 0.1880],LineWidth=2)




%figure settings
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
%pax.RAxis.Label.String = 'Distance [m]';
%title("Sample positions")

pax.RMinorGrid = 'on';
pax.RAxis.MinorTickValues = [0.5, 1.5, 2.5];
rtickformat('%g m')
set(gcf,'Position', [0   0   750   750])

pax.ThetaMinorGrid = 1;
pax.ThetaAxis.MinorTickValues = 15:15:345;
thetatickformat('degrees')

rlim([0,2.75])
