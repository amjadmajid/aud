#Python script to analyse individual recordings, which can cut the raw recordings into individual recordings
#TODO: write the audio scripts to smaller batches

from inspect import stack
from math import cos, sin
from random import sample
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import glob
  
#local test path
localPath = "E:/sampled/"
fileToFind = "rec_050cm_000_locH2-FS.wav"

#method from pytorch that allows the plotting of a waveform
#requires interactive environment (ipynb)
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    print("channel number" + str(num_channels))
    print("frame number" + str(num_frames))

    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)



def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

#calculate x and y coordinates, based on the title of the file
#required filename: rec_xxxcm_yyy_locHz-ww*.wav
# xxx: distance to receiver
# yyy: angle of receiver
# z: name of location (not used)
# ww: type of measurement: [FS/IC/OC]
# TODO: add assertions for file type checks
def loc_to_xy(filename):


    # print("Calculating x and y coordinates")
    distance = int(filename[4:7]) 
    angle = int(filename[10:13])

    measure_type = filename[20:22]

    # print("Found title values " + str(distance) + " and angle " + str(angle))
    # print("measurement type is " + measure_type)

    if (measure_type == "FS"): 
        # print("FS detected")
        #TODO: add calculation for FS measurements
        x = distance * cos(angle)
        y = distance * sin(angle)

        #negative y if condition holds
        if angle < 270 and angle >90:
            y *= -1


        #negative x if condition holds
        if angle > 180:
            x *= -1

        #divide by 250 to properly allow scale
        return x/250,y/250

    elif (measure_type == "IC"):
        print("IC detected")
        #TODO: add calculation for IC measurements

        x =-1
        y = -1

        return x,y

    elif (measure_type == "OC"):
        print("OC detected")
        #TODO: add calculation for OC measurements

        x =-1
        y = -1

        return x,y
    else:
        print("no proper type detected, aborting...")
        exit(-1)



#unpack audio file, add toroidal padding and corresponding coordinates
def read_audio_regression(path, filename):
    # print("Testing")
 
    # metadata = torchaudio.info(path + "\\"+ filename)
    waveform, sample_rate = torchaudio.load(path + "\\"+ filename)
    # print(metadata)

    #add toroidal padding (2 channels before channel 1, which are in order mic channel 5 and 6, 3 after channel 6, channel 1,2,3) to allow full circular audio    
    #create new shape in x, which contains the toroidal padding
    #contains only 0 for now
    
    data = torch.zeros(11, waveform.shape[1])
    
    #copy first 6 channels
    index = torch.tensor([2, 3, 4,5,0,1 ])
    data.index_copy_(0, index, waveform)

    #copy remaining 5 channels
    index = torch.tensor([8, 9, 10,5,6,7 ])
    data.index_copy_(0, index, waveform)

    # print(data.shape)

    # print statistics of the created tensor
    # print_stats(x, sample_rate=sample_rate)

    #plot all values with matplotlib to check proper shifting

    # plot_waveform(x, sample_rate)

    #add label to tensor
    x_coord, y_coord = loc_to_xy(filename)


    # print(data.shape)
    return data[:,:1060], torch.tensor([x_coord, y_coord])

#path to load all audio clips of a certain filetype 
#input: local path of the user to search for the filetypes in
#fileType: type of recordings we want to find
#   None - all files will be used
#   FS   - only FS files will be used
#   IC   - only reverberant files will be used
#   OC   - only Non-Line-Of-Sigth will be used
def read_all_audio_in_dir(path, fileType=None):
    list_of_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if fileType in file:
                list_of_files.append(file)

    #create save tensor
    tensorsFound = []
    labelsFound = []
    for name in list_of_files[:]:
        # print(name)
        datapoint, label = read_audio_regression(path,name)
        # print(label)
        tensorsFound.append(datapoint)
        labelsFound.append(label)
    
    #convert list of tensors to one tensor

    stacked_tensor = torch.stack(tensorsFound)
    stacked_tensor = stacked_tensor.unsqueeze(1)
    labels_tensor = torch.stack(labelsFound)
    return stacked_tensor, labels_tensor


returnedTensor, labels = read_all_audio_in_dir(localPath, "FS")
print(returnedTensor.shape)
# print(returnedTensor)