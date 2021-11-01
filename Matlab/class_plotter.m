clearvars
close all

locs = zeros(60, 2);

for i = 1:length(locs)
    if i <= 60
        locs(i,1) = mod(i-1,12)*30;
        locs(i,2) = ceil(i/12)*20;
    else
        locs(i,1) = mod(i-1-12,24)*15;
        locs(i,2) = (ceil((i-12)/24)-1)*100;
    end
end

locs(:,1) = deg2rad(locs(:,1));

polarscatter(locs(:,1),locs(:,2));
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';


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