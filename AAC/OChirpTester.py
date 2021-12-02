import numpy as np
import time
from OChirpEncode import OChirpEncode
from OChirpDecode import OChirpDecode
import sounddevice as sd
from OChirpOldFunctions import add_wgn
from scipy.io.wavfile import write
from pathlib import Path
import os
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
import os.path
from configuration import Configuration, get_configuration_encoder


def run_orthogonal_test(encoders: list, data_to_send: str, plot: bool = False) -> (float, float):
    """
        Run a test with a list of encoders and what data to send.

        Then we test whether the encoders are orthogonal by transmitting everything at the same time
        (left/right speaker channel)
    """

    decoder1 = OChirpDecode(original_data=data_to_send, encoder=encoders[0])
    decoder2 = OChirpDecode(original_data=data_to_send, encoder=encoders[1])

    filename1, data1 = encoders[0].convert_data_to_sound(data_to_send, filename="temp1.wav")
    filename2, data2 = encoders[1].convert_data_to_sound(data_to_send, filename="temp2.wav")

    # Merge channels
    merged_data = []
    for i in range(len(data1)):
        dual_channel = (data1[i], data2[i])
        merged_data.append(dual_channel)

    merged_data = np.array(merged_data)

    write("test_dual_channel.wav", encoders[0].fsample, merged_data)

    sd.play(merged_data, encoders[0].fsample, blocking=False, mapping=[1, 2])

    # Just record the message
    decoder1.decode_live(plot=False, do_not_process=True)

    # make sure we finished playing (decoder should block though)
    sd.wait()

    ber1 = decoder1.decode_file(file="microphone.wav", plot=plot)
    ber2 = decoder2.decode_file(file="microphone.wav", plot=plot)

    return ber1, ber2


def play_and_record():
    data_to_send = "Hello, World"

    """
        Settings to play with:
            - Chirp range (fs-fe)
            - Symbol time (T)
            - Number of transmitters (M)
            - Blank space
    """

    encoder = OChirpEncode(orthogonal_preamble=True, T_preamble=0.2)
    decoder = OChirpDecode(original_data=data_to_send, encoder=encoder)

    filename, data = encoder.convert_data_to_sound(data_to_send)

    sd.play(data, encoder.fsample, blocking=False)

    decoder.decode_live(plot=True)

    # make sure we finished playing (decoder should block though)
    sd.wait()


def test_orthogonality():
    """
       We want to transmit two supposedly orthogonal symbols and decode them to see if they have no effect on each other
    """
    data_to_send = "Hello, World"

    encoder1 = OChirpEncode(fs=10000, fe=20000, blank_space_time=0.015, f_preamble_start=100,
                            f_preamble_end=7000, orthogonal_pair_offset=0, minimize_sub_chirp_duration=True,
                            required_number_of_cycles=50, M=8, no_window=True)
    encoder2 = OChirpEncode(fs=10000, fe=20000, blank_space_time=0.015, f_preamble_start=100,
                            f_preamble_end=7000, orthogonal_pair_offset=2, minimize_sub_chirp_duration=True,
                            required_number_of_cycles=50, M=8, no_window=True)
    bers = run_orthogonal_test([encoder1, encoder2], data_to_send, plot=True)
    bers = run_orthogonal_test([encoder1, encoder1], data_to_send, plot=True)
    bers = run_orthogonal_test([encoder2, encoder2], data_to_send, plot=True)
    print(bers)


def range_test():
    """
        Iterate over numerous configurations to test them at various distances
        Note: we need to change the distance manually
    """
    data_to_send = "Hello, World"

    base_folder = "./data/results/30-11-2021"

    fs = 5500
    fe = 9500
    M = 8
    configurations_to_test = [
        # Basic configuration
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0, T=0.048, orthogonal_preamble=True, T_preamble=0),
        # Fast basic configuration
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0, T=0.024, orthogonal_preamble=True, T_preamble=0),
        # Tweaked fast basic configuration
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0.012, T=0.024, orthogonal_preamble=True, T_preamble=0.048),
        # Speed
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0.005, T=None, orthogonal_preamble=True, T_preamble=0.048,
                     required_number_of_cycles=10),
    ]
    iterations = 3

    # Give me time to walk away
    time.sleep(10)

    distance = 2
    for i, encoder in enumerate(configurations_to_test):
        Path(f"{base_folder}/{distance}m").mkdir(parents=True, exist_ok=True)
        for j in range(iterations):

            decoder = OChirpDecode(original_data=data_to_send, encoder=encoder)

            filename, data = encoder.convert_data_to_sound(data_to_send)

            # Add some white noise at the beginning and end, to make sure the JBL speaker is initialized and does
            # not stop too early
            z = np.ones(5000)
            z = add_wgn(z, -20)
            data = np.append(z, data)
            data = np.append(data, z)

            sd.play(data, encoder.fsample, blocking=False)

            # Decode live, only used for the mic recording
            decoder.decode_live(plot=False, do_not_process=True)

            # Move and rename the recording
            os.rename("./microphone.wav", f"{base_folder}/{distance}m/{i}_{j}.wav")

            # make sure we finished playing (decoder should block though)
            sd.wait()

            # Ensure any multipath has faded
            time.sleep(0.1)


def get_range_test_results():
    """
        This file is REALLY convoluted, but parses the results from `range_test` and plots them.
    """

    data_send = "Hello, World"

    files = glob(".\\data\\results\\30-11-2021\\**\\*.wav", recursive=True)

    # MAKE SURE THIS IS THE SAME AS WHEN YOU'VE RUN THE TEST
    fs = 5500
    fe = 9500
    M = 8
    configurations_to_test = [
        # Basic configuration
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0, T=0.048, orthogonal_preamble=True, T_preamble=0),
        # Fast basic configuration
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0, T=0.024, orthogonal_preamble=True, T_preamble=0),
        # Tweaked fast basic configuration
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0.012, T=0.024, orthogonal_preamble=True, T_preamble=0.048),
        # Speed
        OChirpEncode(M=M, fs=fs, fe=fe, blank_space_time=0.005, T=None, orthogonal_preamble=True, T_preamble=0.048,
                     required_number_of_cycles=10),
    ]

    print(files)

    def process_file(file: str) -> (OChirpEncode, float, int, int, float):
        print(file)

        filename = file.split("\\")[-1]

        distance = int(file.split("\\")[-2][:-1])
        file_info = filename.split("_")

        # {i}_{j}.wav
        print(file_info)
        config_number = int(file_info[0])
        iteration = int(file_info[1][:-4])

        encoder = configurations_to_test[config_number]
        decoder = OChirpDecode(original_data=data_send, encoder=encoder)

        if distance == 2 and config_number == 2 and False:
            ber = decoder.decode_file(file, plot=True)
        else:
            ber = decoder.decode_file(file, plot=False)
        plt.show()

        return encoder, distance, config_number, iteration, ber

    data_list = []
    for file in files:
        data_list.append(process_file(file))

    df = pd.DataFrame(data_list, columns=["encoder", "distance", "Configuration", "iteration", "ber"])

    # TODO: extract info from encoder and remove encoder column
    # Data such as symbol time to calculate bit rate, maybe just load all parameters
    df = df.drop(columns='encoder')
    df.to_csv("./data/results/30-11-2021/raw_test_results.csv", index=False)

    df = df.groupby(["distance", "Configuration"], as_index=False).ber.agg(['mean', 'std']).reset_index()

    df.to_csv("./data/results/30-11-2021/test_results.csv", index=False)


def test_baseline_configuration(short_symbols: bool = False):
    """
        This configuration presents a simple baseline:
            - No preamble
            - Long symbol size (48ms) (20.8bps)
            - Localization can be done on every chirp
        Since we also want to try 24ms chirps, we can change this with `short_symbols`. (41.7bps)
    """
    data_to_send = "Hello, World!"

    if short_symbols is False:
        encoder = get_configuration_encoder(Configuration.baseline)
    else:
        encoder = get_configuration_encoder(Configuration.baseline_fast)

    decoder = OChirpDecode(original_data=data_to_send, encoder=encoder)

    filename, data = encoder.convert_data_to_sound(data_to_send)

    sd.play(data, encoder.fsample, blocking=False)

    decoder.decode_live(plot=True)

    # make sure we finished playing (decoder should block though)
    sd.wait()


def test_advanced_configuration():
    """
        This configuration presents a more advanced communication:
            - With orthogonal preamble to localize on (48ms)
            - Short symbol size (23.9ms) (41.8bps)
            - Localization can only be done on the preamble
            - Nearly the same as the `short_symbols=True` baseline configuration
                - Comparable bitrate
                - But with preamble
                - And with a shorter active symbol and a longer blank space
                - A compromise with a weaker symbol but better detectable preamble
    """
    data_to_send = "Hello, World!"

    encoder = get_configuration_encoder(Configuration.balanced)

    decoder = OChirpDecode(original_data=data_to_send, encoder=encoder)

    filename, data = encoder.convert_data_to_sound(data_to_send)

    sd.play(data, encoder.fsample, blocking=False)

    decoder.decode_live(plot=True)

    # make sure we finished playing (decoder should block though)
    sd.wait()


if __name__ == '__main__':
    # play_and_record()
    # test_orthogonality()
    # range_test()
    # get_range_test_results()
    # test_baseline_configuration()
    test_advanced_configuration()
