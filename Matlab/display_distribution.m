function display_distribution(database, plot_title)
% show distribution in database
figure

if isa(database.Labels, 'table')
    hist3(double(string(table2array(database.Labels))),'nbins',[12,5])
    xlabel("Direction")
    xticks(0:30:330)
    ylabel("Distance")
    yticks(20:20:100)
else
    histogram(database.Labels)
    if min(double(string(database.Labels))) == 0
        xlabel("Direction")
    else
        xlabel("Distance")
    end
end

title(plot_title)

end