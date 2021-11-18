import numpy as np
import time
from OChirpEncode import OChirpEncode
from OChirpDecode import OChirpDecode
import sounddevice as sd
from OChirpOldFunctions import add_wgn


def run_test(encoder: OChirpEncode, decoder: OChirpDecode, data_to_send: str) -> float:

    filename, data = encoder.convert_data_to_sound(data_to_send)

    sd.play(data, encoder.fsample, blocking=False)

    ber = decoder.decode_live(plot=False)

    # make sure we finished playing (decoder should block though)
    sd.wait()

    return ber


def play_and_record():
    data_to_send = "Hello, World"

    """
        Settings to play with:
            - Chirp range (fs-fe)
            - Symbol time (T)
            - Number of transmitters (M)
            - Blank space
    """

    encoder = OChirpEncode(fs=10000, fe=20000, blank_space_time=0.005, f_preamble_start=0,
                           f_preamble_end=10000, T_preamble=0.250, minimal_sub_chirp_duration=True,
                           required_number_of_cycles=30, M=4)
    decoder = OChirpDecode(original_data=data_to_send, encoder=encoder)

    filename, data = encoder.convert_data_to_sound(data_to_send)

    # Add some white noise at the beginning, to make sure the JBL speaker has initialized
    # z = np.ones(10000)
    # z = add_wgn(z, -60)
    # data = np.append(z, data)
    # from matplotlib import pyplot as plt
    # plt.plot(data)
    # plt.show()

    sd.play(data, encoder.fsample, blocking=False)

    decoder.decode_live(plot=True)

    # make sure we finished playing (decoder should block though)
    sd.wait()


def test_orthogonality():
    from scipy.io.wavfile import write

    """
        We want to transmit two supposedly orthogonal symbols and decode them to see if they have no effect on each other
    """
    data_to_send = "Hello, World"

    """
        Settings to play with:
            - Chirp range (fs-fe)
            - Symbol time (T)
            - Number of transmitters (M)
            - Blank space
    """
    encoder1 = OChirpEncode(fs=1000, fe=5000, blank_space_time=0.025, f_preamble_start=100,
                            f_preamble_end=1000, orthogonal_pair_offset=0, minimal_sub_chirp_duration=True,
                            required_number_of_cycles=15)
    encoder2 = OChirpEncode(fs=1000, fe=5000, blank_space_time=0.025, f_preamble_start=100,
                            f_preamble_end=1000, orthogonal_pair_offset=2, minimal_sub_chirp_duration=True,
                            required_number_of_cycles=15)
    decoder1 = OChirpDecode(original_data=data_to_send, encoder=encoder1)
    decoder2 = OChirpDecode(original_data=data_to_send, encoder=encoder2)

    filename1, data1 = encoder1.convert_data_to_sound(data_to_send, filename="temp1.wav")
    filename2, data2 = encoder2.convert_data_to_sound(data_to_send, filename="temp2.wav")

    # Merge channels
    merged_data = []
    for i in range(len(data1)):
        dual_channel = (data1[i], data2[i])
        merged_data.append(dual_channel)

    merged_data = np.array(merged_data)

    write("test_dual_channel.wav", encoder1.fsample, merged_data)

    sd.play(merged_data, encoder1.fsample, blocking=False, mapping=[1, 2])

    decoder1.decode_live(plot=True)

    # make sure we finished playing (decoder should block though)
    sd.wait()

    decoder2.decode_file(file="microphone.wav", plot=True)


def range_test():
    from pathlib import Path
    import os
    data_to_send = "Hello, World"

    base_folder = "../Audio samples/real-life/14-11-2021_measurements_at_lucan_living_room"

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

        # Non orthogonal chirps
        # M, T, fs, fe, f_p_s, f_p_e
        (1, 0.056, 200, 1200, 1200, 2200),
        (1, 0.056, 200, 2200, 2200, 4200),
        (1, 0.056, 200, 5200, 5200, 10200),
        (1, 0.016, 200, 2200, 2200, 4200),
        (1, 0.026, 200, 5200, 5200, 10200),
    ]
    iterations = 5

    distance = 1
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
                                   minimal_sub_chirp_duration=minimize_sub_chirp_duration,
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
    import pandas as pd
    from glob import glob
    from matplotlib import pyplot as plt

    data_send = "Hello, World"

    files = glob("..\\Audio samples\\real-life\\14-11-2021_measurements_at_lucan_living_room\\**\\*.wav", recursive=True)
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
            T = None
        fs = int(file_info[5])
        fe = int(file_info[6])
        f_p_s = int(file_info[7])
        f_p_e = int(file_info[8])
        iteration = int(file_info[9][:-4])
        print(T)
        print(minimize_sub_chirp_duration)
        encoder = OChirpEncode(fs=fs, fe=fe, blank_space_time=blank_space, f_preamble_start=f_p_s, T=T,
                               f_preamble_end=f_p_e, T_preamble=0.250, minimal_sub_chirp_duration=minimize_sub_chirp_duration,
                               required_number_of_cycles=num_cycles, M=M)
        decoder = OChirpDecode(original_data=data_send, encoder=encoder)

        if T is None:
            T = encoder.T

        ber = decoder.decode_file(file, plot=False)
        plt.show()

        return distance, M, blank_space, minimize_sub_chirp_duration, num_cycles, T, fs, fe, f_p_s, f_p_e, iteration, ber

    data_list = []
    for file in files:
        data_list.append(process_file(file))
    df = pd.DataFrame(data_list, columns=["distance", "M", "blank_space", "minimzed_subchirp_duration", "cycles", "T", "fs", "fe", "f_p_s", "f_p_e", "iteration", "ber"])
    pd.options.display.width = 0
    print(df[df.ber > 0])
    df = df.groupby(["distance", "M", "blank_space", "minimzed_subchirp_duration", "cycles", "T", "fs", "fe", "f_p_s", "f_p_e", "iteration"], as_index=False).ber.agg(['mean', 'std']).reset_index()

    df.plot(x="cycles", y="mean", yerr="std", label="ber")
    plt.show()

    print(df)


if __name__ == '__main__':
    play_and_record()
    # test_orthogonality()
    # range_test()
    # get_range_test_results()
