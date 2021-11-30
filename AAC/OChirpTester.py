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

    base_folder = "../Audio samples/real-life/22-11-2021_measurements_at_lucan_living_room"

    """
        For all configs:
            fs=10000, fe=20000, f_preamble_start=1, f_preamble_end=10000, T_preamble=0.250,
            M=[2,4,8]
            iterations=5        
        
        Configs to test:
        
            SPEED (2.8ms):
                 blank_space_time=0.00, minimal_sub_chirp_duration=True, required_number_of_cycles=10,
            
            SPEED COMPROMISE (11.9ms):
                 blank_space_time=0.005, minimal_sub_chirp_duration=True, required_number_of_cycles=30
                               
            ROBUST COMPROMISE (32.8ms):
                 blank_space_time=0.015, minimal_sub_chirp_duration=False, required_number_of_cycles=50
                               
            ROBUST (62ms):
                 blank_space_time=0.06, minimal_sub_chirp_duration=False, required_number_of_cycles=75
                 
    """

    configurations_to_test = [
        (2, 0, True, 10),
        (2, 0.005, True, 30),
        (2, 0.015, False, 50),
        (2, 0.06, False, 75),
        (4, 0, True, 10),
        (4, 0.005, True, 30),
        (4, 0.015, False, 50),
        (4, 0.06, False, 75),
        (8, 0, True, 10),
        (8, 0.005, True, 30),
        (8, 0.015, False, 50),
        (8, 0.06, False, 75),

        # Non orthogonal chirps (M=1)
        # M, T, fs, fe, f_p_s, f_p_e
        (1, 0.056, 200, 1200, 1200, 2200),
        (1, 0.056, 200, 2200, 2200, 4200),
        (1, 0.056, 200, 5200, 5200, 10200),
        (1, 0.016, 200, 2200, 2200, 4200),
        (1, 0.026, 200, 5200, 5200, 10200),
    ]
    iterations = 5

    # Give me time to walk away
    time.sleep(10)

    distance = 0
    for config in configurations_to_test:
        for i in range(iterations):
            M = int(config[0])

            if M != 1:
                blank_space = float(config[1])
                minimize_sub_chirp_duration = bool(config[2])
                num_cycles = int(config[3])
                T = None
                fs = 10000
                fe = 20000
                f_p_s = 0
                f_p_e = 7000
            else:
                blank_space = 0.006
                minimize_sub_chirp_duration = False
                num_cycles = 5
                T = float(config[1])
                fs = int(config[2])
                fe = int(config[3])
                f_p_s = int(config[4])
                f_p_e = int(config[5])

            encoder = OChirpEncode(fs=fs, fe=fe, blank_space_time=blank_space, f_preamble_start=f_p_s,
                                   f_preamble_end=f_p_e, T_preamble=0.250, T=T,
                                   minimize_sub_chirp_duration=minimize_sub_chirp_duration,
                                   required_number_of_cycles=num_cycles, M=M)
            decoder = OChirpDecode(original_data=data_to_send, encoder=encoder)

            print(f"Testing M=[{M}] blank space=[{blank_space}] num cycles = [{num_cycles}]")
            Path(f"{base_folder}/{distance}m").mkdir(parents=True, exist_ok=True)

            filename, data = encoder.convert_data_to_sound(data_to_send)

            # Add some white noise at the beginning and end, to make sure the JBL speaker is initialized and does
            # not stop too early
            z = np.ones(2500)
            z = add_wgn(z, -40)
            data = np.append(z, data)
            data = np.append(data, z)

            sd.play(data, encoder.fsample, blocking=False)

            # Decode live, only used for the mic recording
            decoder.decode_live(plot=False, do_not_process=True)

            # Move and rename the recording
            os.rename("./microphone.wav", f"{base_folder}/{distance}m/{M}_{blank_space}_{int(minimize_sub_chirp_duration)}_{num_cycles}_{T}_{fs}_{fe}_{f_p_s}_{f_p_e}_{i}.wav")

            # make sure we finished playing (decoder should block though)
            sd.wait()

            # Ensure any multipath has faded
            time.sleep(0.1)


def get_range_test_results():
    """
        This file is REALLY convoluted, but parses the results from `range_test` and plots them.
    """

    data_send = "Hello, World"

    files = glob("..\\Audio samples\\real-life\\22-11-2021_measurements_at_lucan_living_room\\**\\*.wav", recursive=True)
    print(files)

    def process_file(file: str) -> (int, int, float, bool, int, int, float):
        print(file)

        filename = file.split("\\")[-1]

        distance = int(file.split("\\")[-2][:-1])
        file_info = filename.split("_")

        # {M}_{blank_space}_{int(minimize_sub_chirp_duration)}_{num_cycles}_{T}_{fs}_{fe}_{f_p_s}_{f_p_e}_{i}.wav
        print(file_info)
        M = int(file_info[0])
        blank_space = float(file_info[1])
        minimize_sub_chirp_duration = bool(int(file_info[2]))
        num_cycles = int(file_info[3])
        try:
            T = float(file_info[4])
        except ValueError:
            # If we minimize the cycles
            T = None
        fs = int(file_info[5])
        fe = int(file_info[6])
        f_p_s = int(file_info[7])
        f_p_e = int(file_info[8])
        iteration = int(file_info[9][:-4])
        print(T)
        print(minimize_sub_chirp_duration)
        encoder = OChirpEncode(fs=fs, fe=fe, blank_space_time=blank_space, f_preamble_start=f_p_s, T=T,
                               f_preamble_end=f_p_e, T_preamble=0.250, minimize_sub_chirp_duration=minimize_sub_chirp_duration,
                               required_number_of_cycles=num_cycles, M=M)
        decoder = OChirpDecode(original_data=data_send, encoder=encoder)

        if T is None:
            T = encoder.T

        if M > 1:
            plot = False
        else:
            plot = False

        T = round(T, 5)
        ber = decoder.decode_file(file, plot=plot)
        plt.show()

        return distance, M, blank_space, minimize_sub_chirp_duration, num_cycles, T, fs, fe, f_p_s, f_p_e, iteration, ber

    if not os.path.isfile("test_results.csv"):
        data_list = []
        for file in files:
            if file.split("\\")[-1][0].isdigit():
                data_list.append(process_file(file))

        df = pd.DataFrame(data_list, columns=["distance", "M", "blank_space", "minimzed_subchirp_duration", "cycles", "T", "fs", "fe", "f_p_s", "f_p_e", "iteration", "ber"])

        df["Configuration"] = df.apply(lambda row: "-".join([str(i) for i in row[1:-2]]), axis=1, raw=True)
        df["Threshold_scheme"] = ""
        tempdf = df[["Configuration", "Threshold_scheme", "distance", "ber", "iteration"]]
        tempdf = tempdf.rename(columns={"distance": "Distance", "iteration": "Iteration", "ber": "BER"})
        tempdf.to_csv("../Results/real-life-tests/chirps.csv", index=False)

        df.to_csv("raw_test_results.csv", index=False)

        df = df.groupby(["distance", "M", "blank_space", "minimzed_subchirp_duration", "cycles", "T", "fs", "fe", "f_p_s", "f_p_e"], as_index=False).ber.agg(['mean', 'std']).reset_index()

        df.to_csv("test_results.csv", index=False)
    else:
        # df = pd.read_csv("test_results.csv")
        df = pd.read_csv("raw_test_results.csv")

    pd.options.display.width = 0
    print(df)

    # Some boxplot settings
    color_list = ["#a8ddb5", '#7bccc4', '#43a2ca', '#0868ac', '#eff3ff', '#0000ff']
    medianprops = {'color': color_list[3], 'linewidth': 2}
    boxprops = {'color': color_list[3], 'linestyle': '-'}
    whiskerprops = {'color': color_list[3], 'linestyle': '-'}
    capprops = {'color': color_list[3], 'linestyle': '-'}

    # Plot average difference per M
    plt.figure(figsize=(6, 3))
    df = df[df.M != 1]
    index = 0
    labels = []
    for distance in df.distance.unique():
        for m in df.M.unique():
            print(f"{distance} {m}")
            data = df[(df.M == m) & (df.distance == distance)]

            plt.boxplot(data['ber'], positions=[index], showfliers=False, medianprops=medianprops, boxprops=boxprops,
                        whiskerprops=whiskerprops, capprops=capprops, widths=0.65)
            # hardcoded to be at the middle on the x-axis
            if (index + 2) % 3 == 0:
                plt.text(x=index - 1, y=0.7, s=f"{distance} meters", color='black')

            labels.append(f"{m}")
            index += 1
            plt.axvline(x=index - 0.5, color='black', alpha=0.2)
            # plt.text(x=index-1, y=-0.025, s=m, color='r')
        if distance < 5:
            plt.axvline(x=index-0.5, color='r', linestyle="dashed")

    plt.ylabel("BER")
    plt.xlabel("M")
    plt.xticks(np.arange(index), labels)
    plt.tight_layout()
    plt.show()

    # Try to plot all data...
    plt.figure(figsize=(8.27*2, 6))
    index = 0
    labels = []
    plt.text(x=-2.5, y=-0.025, s='m=', color='r')
    for distance in df.distance.unique():
        for m in df.M.unique():
            data = df[(df.M == m) & (df.distance == distance)]
            gb = data.groupby(["blank_space", "minimzed_subchirp_duration", "cycles", "T", "fs", "fe", "f_p_s", "f_p_e"])

            for name, group in gb:
                # Drop bad configs
                good_configs = [(0.015, False, 50)]

                # good = np.any([item[0] == name[0] and item[1] == name[1] and item[2] == name[2] for item in good_configs])
                good = True
                if good:
                    print(name)
                    plt.boxplot(group['ber'], positions=[index], showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)

                    # hardcoded to be at the middle on the x-axis
                    if (index + 9) % 17 == 0:
                        labels.append(distance)
                    else:
                        labels.append("")
                    index += 1

            plt.axvline(x=index - 0.5, color='black', alpha=0.2)
            plt.text(x=index-3, y=-0.025, s=m, color='r')

        if distance < 5:
            plt.axvline(x=index-0.5, color='r', linestyle="dashed")

    plt.xticks(np.arange(index), labels)
    plt.ylabel("ber")
    plt.xlabel("distance [m]")

    plt.show()


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
        symbol_time = 0.048
    else:
        symbol_time = 0.024

    encoder = OChirpEncode(T=symbol_time, T_preamble=0)
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

    encoder = OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, blank_space_time=0.01,
                           required_number_of_cycles=10, minimize_sub_chirp_duration=False)
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
