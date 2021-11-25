clearvars
close all

locs = zeros(120, 2);

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

polarscatter(locs(:,1),locs(:,2),'filled');
hold on
polarscatter(0,0,500,[0.4660 0.6740 0.1880],'filled');

pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
%pax.RAxis.Label.String = 'Distance [m]';
%title("Sample positions")

pax.RMinorGrid = 'on';
pax.RAxis.MinorTickValues = [0.5, 1.5, 2.5];
rtickformat('%g m')
set(gcf,'Position', [226   122   654   749])

pax.ThetaMinorGrid = 1;
pax.ThetaAxis.MinorTickValues = 15:15:345;
thetatickformat('degrees')

%voronoi
[x, y] = pol2cart(locs(:,1),locs(:,2));
[v,c] = voronoin([x y]) ;
figure
hold on
voronoi(x,y)
A = zeros(length(c),1) ;
for i = 1:length(c)
    v1 = v(c{i},1) ; 
    v2 = v(c{i},2) ;
    patch(v1,v2,rand(1,3))
    A(i) = polyarea(v1,v2) ;
end

pbaspect([1 1 1])
xlim([-3.5, 3.5]);
ylim([-3.5, 3.5]);