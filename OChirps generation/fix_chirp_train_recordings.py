from glob import glob
from pathlib import Path
from scipy.io.wavfile import read, write
# from AAC.OChirpEncode import OChirpEncode
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

    stds = []
    for channel in range(number_of_channels-2):
        channel_data = data[:, channel]
        stds.append(np.std(channel_data))

    write(destination, fs, data)


def reorder_channels():
    Path(ordered_files).mkdir(parents=True, exist_ok=True)

    files = glob(original_location + '/**/*.wav', recursive=True)
    print(files)

    with Pool(12) as p:
        p.map(reorder_channel, files)


def main():
    reorder_channels()


if __name__ == '__main__':
    main()
