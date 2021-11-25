function PI_audio()
%#codegen
rpi = raspi();

%prealocate record to padded size
rec_len = 22050;

rec_steps = 100;    % cut recording in
frame_size = ceil(rec_len/rec_steps);
tot_rec_len = frame_size*rec_steps;


if tot_rec_len > rec_len
    startpoint = floor((tot_rec_len - rec_len)/2);
else
    startpoint = 1;
end


num_channels = 6;

l_pad = 2;
r_pad = 3;

rec_buff = zeros(tot_rec_len,num_channels,'int16');
rec = zeros(rec_len,num_channels+l_pad+r_pad,'int16');

fprintf("setup audio capture\n")
%TODO: find a way to automatically detect that the capture device setup
%went wrong/no recording
cap_obj = audiocapture(rpi,'plughw:2,0', 'SampleRate', 44100, ...
    'SamplesPerFrame', frame_size, 'NumberOfChannels', num_channels); %mic array


%load neural network
fprintf("load nn\n")
net        = coder.loadDeepLearningNetwork('Audio_loc_net.mat');
[~,~] = net.classify(zeros(rec_len,num_channels+l_pad+r_pad)');
fprintf("net loaded\n")




fprintf("Running PI_audio echo test.\n")

for k = uint16(1:3000)
    tic;

    fprintf("%u\ncapture\n",k)

    [~] = capture(cap_obj);  % sacreficial capture

    for i = 1:rec_steps
        rec_buff(1+(i-1)*rec_steps:i*rec_steps,:) = capture(cap_obj);
    end

    if tot_rec_len > rec_len % cutoff excess
        startpoint = floor((tot_rec_len - rec_len)/2);
        rec_buff(1:startpoint,:) = [];
        rec_buff(rec_len + 1: end,:) = [];
    end

    r = rms(rec_buff(:,1));
    if r < 10
        fprintf("capture failure, rms = %.3f\n",r)
    else
        fprintf("capture succes, rms = %.3f\n", r)
    end

    fprintf("Capture time: %.2f s\n",toc);

    rec(:,1+l_pad: end-r_pad) = rec_buff(startpoint: startpoint + rec_len -1,:);

    %preprocess

    rec = toroidal_padding(rec,0,0,l_pad,r_pad);

    % classify

    fprintf("Clasiffying\n")
    [label,score] = net.classify(rec');
    maxScore = max(score);

    labelStr = cellstr(label);
    fprintf('Label : %s \nScore : %f\nDuration: %.2f seconds\n\n',labelStr{:},maxScore,toc);
end
end

function X = toroidal_padding(X, t, b, l, r)
% Version without matrix multiplication

if t~=0
    for i = 0:t-1
        X(t-i,:) = X(end-b-i,:);
    end
end

if b ~=0
    for i = 1:b
        X(end-b+i,:) = X(t+i,:);
    end
end

if l~=0
    for i = 0:l-1
        X(:,l-i) = X(:,end-r-i);
    end
end

if r ~=0
    for i = 1:r
        X(:,end-r+i) = X(:,l+i);
    end
end

end