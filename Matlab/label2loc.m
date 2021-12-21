function [r, theta] = label2loc(label)
r = str2double(label(8:10));
theta = str2double(label(1:3));
end