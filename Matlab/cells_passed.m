function num = cells_passed(c, v, x_start, y_start, x_end, y_end,plot_flag)
%returns the number of voronoi cells that the line from (x_start,y_start)
%to (x_end, y_end) passes trough

num = -1;

x_line = linspace(x_start, x_end)';
y_line = linspace(y_start, y_end)';


for i = 1:length(c)
    if size(c{i}, 1) > size(c{i},2)
        c{i} = c{i}';
    end

    vx = v([c{i}, c{i}(1)],1) ;
    vy = v([c{i}, c{i}(1)],2) ;

    if plot_flag
        figure(999)
        plot (vx, vy,x_line,y_line)
        xlim([-300,300])
        ylim([-300,300])
    end


    [in, on] = inpolygon(x_line, y_line,vx, vy);

    if numel(x_line(in)) > numel(x_line(on))
        num = num+1;
    end

end


end