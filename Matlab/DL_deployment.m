clearvars -except rpi

if ~(exist('rpi','var') && isa(rpi,'raspi'))
    rpi = raspi('Robomindpi-002','pi',input("PI Password:\n",'s'));
end

board = targetHardware('Raspberry Pi');
board.CoderConfig.TargetLang = 'C++';

dlcfg = coder.DeepLearningConfig('arm-compute');
dlcfg.ArmArchitecture = 'armv7';

% input arm version, use line below to find it for the specific pi
%r.system('strings $ARM_COMPUTELIB/lib/libarm_compute.so | grep arm_compute_versio | cut -d\  -f 1')
dlcfg.ArmComputeVersion = '20.02.1';

board.CoderConfig.DeepLearningConfig = dlcfg;
deploy(board, 'PI_audio.m')

% devices = listAudioDevices(rpi,'playback');