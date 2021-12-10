from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from AAC.OChirpEncode import OChirpEncode
from AAC.OChirpDecode import OChirpDecode
from multiprocessing import Pool
import numpy as np
import os

original_location = '../Recorded_files/'
ordered_files = '../Ordered_files/'
final_files = '../Sampled_files/'


def reorder_channel(chirp_train: str):
    
    destination = chirp_train.replace(original_location[:-1], ordered_files[:-1])

    # Skip the file if it exists
    if os.path.isfile(destination):
        return
    else:
        Path(os.path.split(destination)[0]).mkdir(parents=True, exist_ok=True)

    print(chirp_train)

    fs, data = read(chirp_train)

    number_of_channels = data.shape[-1]

    if number_of_channels != 8:
        print(f"ERROR: not 8 channels in this audio file: [{chirp_train}]")
        return

    stds = []
    for channel in range(number_of_channels):
        channel_data = data[:, channel]
        stds.append(np.std(channel_data))

    first_empty_channel = np.argmin(stds)
    stds.pop(first_empty_channel)
    second_empty_channel = np.argmin(stds)
    stds.pop(second_empty_channel)

    max_channel_nr = number_of_channels-1
    if first_empty_channel != max_channel_nr and second_empty_channel != max_channel_nr:
        high_index = np.max([first_empty_channel, second_empty_channel])
        low_index = np.min([first_empty_channel, second_empty_channel])

        begin = np.arange(high_index + 1, max_channel_nr + 1)
        end = np.arange(0, low_index)

        selected_channels = np.append(begin, end)
    else:
        selected_channels = np.arange(0, np.min([first_empty_channel, second_empty_channel]))

    data = data[:, selected_channels]

    write(destination, fs, data)


def reorder_channels():

    wav_files = glob(original_location + '/**/*.wav', recursive=True)

    #filter out non chirp train recordings
    files = [file for file in wav_files if not "z_copy" in file and "chirp_train" in file]
    
    #print(files)
    print("{} files found".format(len(files)))

    with Pool(12) as p:
        p.map(reorder_channel, files)


def generate_sample(chirp_train: str):
    #print(chirp_train)

    placeholder_marker = "PLACEHOLDER_INT"

    destination = chirp_train.replace(ordered_files[:-1], final_files[:-1])
    destination = destination.replace("Raw_recordings", "Samples")
    destination = destination.replace(".wav", f"_s{placeholder_marker}.wav")

    # Cannot check if the file exist here, since we have 200 files to check.

    def get_symbol_period(chirp_train: str) -> float:
        if "0s024" in chirp_train:
            return 0.024
        elif "0s048" in chirp_train:
            return 0.048
        print("ERROR, no known time in: {}".format(chirp_train))

    def get_chirp_index(chirp_train: str) -> int:
        if "_0\\" in chirp_train:
            return 0
        elif "_1\\" in chirp_train:
            return 1
        elif "_2\\" in chirp_train:
            return 2
        elif "_3\\" in chirp_train:
            return 3
        elif "_4\\" in chirp_train:
            return 4
        elif "_5\\" in chirp_train:
            return 5
        elif "_6\\" in chirp_train:
            return 6
        elif "_7\\" in chirp_train:
            return 7

    # Get chirp info
    T = get_symbol_period(chirp_train)
    index = get_chirp_index(chirp_train)

    # Read audio
    fsample, data = read(chirp_train)

    # used for finding the peaks
    reference_data = data[:, 0]

    encoder = OChirpEncode(T=T+0.1, T_preamble=0.25, blank_space_time=0.1, f_preamble_start=1, f_preamble_end=5500,
                           orthogonal_pair_offset=index, fsample=fsample)
    decoder = OChirpDecode(encoder=encoder, original_data=chr(0x00) * 25)

    symbols = decoder.get_symbols(no_window=False)

    # Find the index of the preamble, such that we can check whether we detect peaks before this (incorrect)
    preamble_index = decoder.contains_preamble(reference_data, preamble_index=True, threshold_multiplier=10)

    # Convolve the data with the original symbols, this produces periodic peaks
    conv_data = decoder.get_conv_results(reference_data, symbols)

    # Find the peaks
    peaks = decoder.get_peaks(conv_data, plot=False, N=len(decoder.original_data_bits))
    peaks = np.array(list(map(lambda x: x[0], peaks)))

    # Remove any peaks that occur before the preamble
    peaks = peaks[peaks > preamble_index]

    # How wide should we select the sample?
    # Can tweak this is we don't want any overlap
    sample_width = encoder.T * fsample

    # How much should we add to the peak such that it shifts to the right
    # We use this to make sure that we center the sample nicely.
    sample_offset = sample_width * 0.2

##    # plotting
##    fig, axs = plt.subplots(6, sharex=True, sharey=True)
##    for j in range(6):
##        dataset = data[:, j]
##        ax = axs[j]
##        ax.plot(dataset)
##
##    for i, peak in enumerate(peaks):
##        start = int(sample_offset + peak - sample_width/2)
##        end = int(sample_offset + peak + sample_width/2)
##        
##        ax.vlines(start, np.min(dataset), np.max(dataset), color="black", alpha=0.5)
##        ax.vlines(end, np.min(dataset), np.max(dataset), color="black", alpha=0.5)
##
##    plt.tight_layout()
##    plt.show()
        

    # Sanity checks
    correct_peak_diff = 5474 if T == 0.024 else 6530
    peak_diff = np.diff(peaks)
    mean_peak_diff = np.mean(peak_diff)
    correct_heights_std = 100000
    heights_std = np.std(conv_data[0][np.array(peaks)])
    peak_diff_std_threshold = 200
    if len(peaks) != 200 or \
            np.abs(mean_peak_diff - correct_peak_diff) > (sample_width * 0.005) or \
            np.std(peak_diff) > peak_diff_std_threshold or \
            heights_std > correct_heights_std:
        print(f"ERROR! {chirp_train}\nWe cannot decode this file! mean peak diff: {mean_peak_diff}, but should be {correct_peak_diff}\n"
              f"or the avg peak interval differs too much: {np.abs(mean_peak_diff - correct_peak_diff)} > {(sample_width * 0.005)}\n"
              f"or there is too much variance in the typical peak dfference: {np.std(peak_diff)} > {peak_diff_std_threshold}\n"
              f"or there are not enough peaks {len(peaks)} != 200\n"
              f"or the height of the peaks is inconsistent: {heights_std} > {correct_heights_std}\n")

        # Plot some results to show the issues
        decoder.get_peaks(conv_data, plot=True, N=len(decoder.original_data_bits))
        plt.figure()
        plt.scatter(peaks, conv_data[0][np.array(peaks)], color="red", marker='X', zorder=5)
        plt.plot(conv_data[0])
        plt.show()
        return

    
    for i, peak in enumerate(peaks):
        start = int(sample_offset + peak - sample_width/2)
        end = int(sample_offset + peak + sample_width/2)
         
        sample = data[start:end]

        real_destination = destination.replace(placeholder_marker, str(i).zfill(3))

        # Skip the file if it exists
        #if not os.path.isfile(real_destination):
        Path(os.path.split(destination)[0]).mkdir(parents=True, exist_ok=True)
            # print(f"writing {real_destination}")
        write(real_destination, fsample, sample)
        # else:
        #     print(f"skipping {real_destination}")



def generate_samples():

    wav_files = glob(ordered_files + '/**/*.wav', recursive=True)

    #filter out non chirp train recordings
    files = [file for file in wav_files if not "z_copy" in file and "chirp_train" in file]
    
    #print(files)
    print("{} files found".format(len(files)))

    for file in files:
        generate_sample(file)
        return

    # with Pool(12) as p:
    #     p.map(generate_sample, files)


def main():
    print("Reordering")
    reorder_channels()

    print("Sample_generation")
    generate_samples()
    

if __name__ == '__main__':
    main()
