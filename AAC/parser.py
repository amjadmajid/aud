from glob import glob
from configuration import Configuration, get_configuration_encoder
from OChirpDecode import OChirpDecode
from OChirpEncode import OChirpEncode
import pandas as pd
import os
import numpy as np
from scipy.io.wavfile import read
from multiprocessing import Pool, freeze_support

directory = './data/results/05-01-2022-multi-transmitter-nlos/'
configurations = ['baseline', 'optimized', 'baseline48']
chirp_pair_offsets = [0, 2, 4, 6]

all_configs = []
for c in configurations:
    for o in chirp_pair_offsets:
        all_configs.append(c + str(o))


def calculate_ber_multi_transmitter_async(settings):
    file = settings[0]
    number_of_transmitters = settings[1]
    possible_files = settings[2]
    decoder = settings[3]

    return calculate_ber_multi_transmitter(file, number_of_transmitters, possible_files, decoder)


def calculate_ber_multi_transmitter(file, number_of_transmitters, possible_files, decoder):
    print(f"{number_of_transmitters} transmitters")

    random_files_selection = possible_files.copy()
    np.random.shuffle(random_files_selection)

    # Select number_of_transmitters random files from this array (no redraws)
    selected_files = []
    for i in range(number_of_transmitters):
        selected_files.append(random_files_selection[i][np.random.randint(0, len(random_files_selection[i]))])
        random_files_selection[i].remove(selected_files[-1])

    # Read data from extra transmitters and superimpose it on the original data
    _, original_data = read(file)
    original_data = original_data.astype(np.float64)
    for selected_file in selected_files:

        _, data = read(selected_file)
        data = data.astype(np.float64)
        offset = np.random.randint(0.25 * 44100, original_data.size)  # - (original_data.size // 2)
        if offset < 0:
            offset = 0
        original_data[offset:] = original_data[offset:] + data[offset:]

    original_data = original_data.astype(np.int16)

    # Get the channel with the highest energy
    # print(original_data.shape)
    if len(original_data.shape) > 1 and original_data.shape[1] > 1:
        original_data = original_data[:, np.argmax(np.max(original_data, axis=0), axis=0)]
    # print(original_data.shape)

    # decode the data
    ber = decoder.decode_data(original_data, plot=False)
    # if ber > 0.0:
    #     print("INFORMATION:")
    #     print(f"Ts: {decoder.T}")
    #     print(f"Number of transmitters {number_of_transmitters}")
    #     print(file)
    #     print(selected_files)
    #     np.savetxt(f'saved_array.csv', original_data, delimiter=',')
    #     decoder.decode_data(original_data, plot=True)
    return ber


def parse(n_extra_transmitters: int = 0):
    files = glob(directory + '**/*.wav', recursive=True)

    # We randomly superimpose other transmitters on this one, so we need to make sure that we randomize it a bit
    # So this is how often we repeat (randomly) every file/transmitter combination
    extra_transmitters_iterations = 5

    bers = []

    for file in files:
        print(file)

        conf = None
        offset = None
        config_name = None
        for c in all_configs:
            if '\\' + c + '\\' in file:
                conf = c[:-1]
                offset = int(c[-1])
                config_name = c

        conf = Configuration[conf]
        encoder = get_configuration_encoder(conf)
        encoder.orthogonal_pair_offset = offset
        decoder = OChirpDecode(encoder=encoder, original_data="UUUU")
        filename = os.path.split(file)[1]
        file_path = os.path.split(file)[0]

        distance = int(filename.split('_')[1].replace('cm', ''))

        # if distance != 250:
        #     continue

        if n_extra_transmitters == 0:
            ber = decoder.decode_file(file, plot=False)
            bers.append((conf, distance, 1, ber))
        else:
            # Move the pool to here, otherwise we reconstruct too often
            # with Pool(8) as p:
            ber = decoder.decode_file(file, plot=False)
            bers.append((conf, distance, 1, ber))

            # Get possible configs that may operate as noise
            possible_configs = all_configs.copy()
            print(config_name)
            possible_configs.remove(config_name)

            # Get all valid noise files
            random_files_selection = []
            for possible_config in possible_configs:
                if conf.name in possible_config and (('_' in conf.name) == ('_' in possible_config))\
                        and (('48' in conf.name) == ('48' in possible_config)):
                    new_file_path = file_path.replace(config_name, possible_config)
                    random_files_selection.append(glob(new_file_path + '\\*.wav', recursive=False))

            for extra_transmitters in range(0, n_extra_transmitters+1):
                # settings = []
                # for _ in range(extra_transmitters_iterations):
                #     settings.append((file, extra_transmitters, random_files_selection, decoder))
                #
                # # Make sure we average the randomness of the noise
                # res = p.map(calculate_ber_multi_transmitter_async, settings)
                # for r in res:
                #     bers.append((conf, distance, extra_transmitters, r))

                for _ in range(extra_transmitters_iterations):
                    ber = calculate_ber_multi_transmitter(file, extra_transmitters, random_files_selection, decoder)
                    bers.append((conf, distance, extra_transmitters, ber))

    df = pd.DataFrame(bers, columns=['Configuration', 'distance', 'transmitters', 'ber'])

    df.to_csv(directory + 'parsed_results_all48.csv', index=False)
    df[df.Configuration != "baseline48"].to_csv(directory + 'parsed_results_all.csv', index=False)

    print(f"avg ber: {df.ber.mean()}")


# def parse_subchirp_file_mt(file: str, transmittor_count: int, files: list) -> (str, int, float, float, int, float):
#     from MQTT_AAC_subchirp_test import get_cycles
#
#     if "\\fixed\\" in file or "/fixed/" in file:
#         conf: str = "fixed"
#     else:
#         conf: str = "dynamic"
#
#     offset: int = int(file.split("_")[-4][-1])
#     Ts: float = float(file.split("_")[-3])
#     cycles_: float = float(file.split("_")[-2])  # Seems to be incorrect
#     cycles: float = get_cycles({"configuration": conf, "symbol_time": Ts, "fstart": 5500, "fend": 9500})
#
#     # if conf != "fixed" or Ts != 0.024:
#     #     return None, None, None, None, None, None
#
#     encoder = OChirpEncode(T=None, fs=5500, fe=9500, T_preamble=0, orthogonal_pair_offset=offset,
#                            minimize_sub_chirp_duration=("dynamic" in conf),
#                            required_number_of_cycles=cycles)
#
#     decoder = OChirpDecode(encoder=encoder, original_data="UUUU")
#     decoder.preamble_min_peak = 1
#
#     print(f"{file} {conf} {offset} {cycles} {transmittor_count}")
#
#     _, original_data = read(file)
#     # Get the channel with the highest energy
#     if len(original_data.shape) > 1 and original_data.shape[1] > 1:
#         original_data = original_data[:, np.argmax(np.max(original_data, axis=0), axis=0)]
#     # Add extra noise
#     if transmittor_count > 1:
#
#         # Get possible configs that may operate as noise
#         possible_noise_files = files.copy()
#         possible_noise_files.remove(file)
#
#         # Get all valid noise files
#         random_files_selection = [[], [], [], []]
#         for possible_config in possible_noise_files:
#             possible_config_offset: int = int(possible_config.split("_")[-4][-1])
#             possible_config_Ts: float = float(possible_config.split("_")[-3])
#             if conf in possible_config and \
#                     possible_config_Ts == Ts and \
#                     possible_config_offset != offset:
#                 random_files_selection[possible_config_offset//2].append(possible_config)
#
#         # print(random_files_selection)
#
#         random_files_selection.remove([])
#
#         # Some samples are incorrect and don't have any possible random selections
#         # So just return None. However, I think I removed all of them.
#         if len(random_files_selection[0]) == 0:
#             return None, None, None, None, None, None
#
#         # Randomize the selection
#         for off in range(3):
#             np.random.shuffle(random_files_selection[off])
#         np.random.shuffle(random_files_selection)
#
#         # Select number_of_transmitters random files from this array (no redraws)
#         selected_files = []
#         for i in range(transmittor_count-1):
#             end = len(random_files_selection[i])
#             if end == 0:
#                 print(f"END IS 0!! {i} {selected_files}")
#             selected_files.append(random_files_selection[i][np.random.randint(0, end)])
#
#         # Construct the noised signal
#         original_data = original_data.astype(np.float64)
#
#         for selected_file in selected_files:
#             _, data = read(selected_file)
#             data = data.astype(np.float64)
#             # Get the channel with the highest energy
#             if len(data.shape) > 1 and data.shape[1] > 1:
#                 data = data[:, np.argmax(np.max(data, axis=0), axis=0)]
#
#             end = min(original_data.size, data.size)
#             random_offset = np.random.randint(0.25 * 44100, original_data.size-1)  # - (original_data.size // 2)
#
#             # print(f"{random_offset} {end} {original_data[random_offset:].shape} {data[random_offset:].shape}")
#
#             print(f"{file} {selected_file}")
#             original_data[random_offset:] = original_data[random_offset:] + data[:end-random_offset]
#
#         original_data = original_data.astype(np.int16)
#
#     ber = decoder.decode_data(original_data, plot=False)
#
#     # if ber > 0 and Ts > 0.015 and Ts != 0.023:
#     #     ber = decoder.decode_data(original_data, plot=True)
#
#     # conf, offset, cycles, ber
#     return conf, offset, Ts, cycles, transmittor_count, ber
#
#
# def parse_subchirp_test():
#     from itertools import repeat
#     freeze_support()
#
#     dir = './data/results/06-01-2022-dynamic-vs-fixed-sub-chirps/Recorded_files/'
#     # dir = 'C:/Users/lucan/Desktop/temp/11-02-2022/'
#     # dir = 'E:/Recorded_files/'
#     files = glob(dir + '**/*.wav', recursive=True)
#
#     transmitters = [1]
#     repeats = 1
#
#     # results = []
#     # for file in files:
#     #     for n in transmitters:
#     #         if n > 1:
#     #             for _ in range(repeats):
#     #                 results.append(parse_subchirp_file_mt(file, n, files))
#     #         else:
#     #             results.append(parse_subchirp_file_mt(file, n, files))
#
#     results = []
#     with Pool(10) as p:
#         for n in transmitters:
#             if n > 1:
#                 for _ in range(repeats):
#                     results.extend(p.starmap(parse_subchirp_file_mt, zip(files, repeat(n), repeat(files))))
#             else:
#                 results.extend(p.starmap(parse_subchirp_file_mt, zip(files, repeat(n), repeat(files))))
#
#     df = pd.DataFrame(results, columns=["configuration", "offset", "Ts", "cycles", "transmitters", "ber"])
#     print(df)
#     df.to_csv(dir + 'parsed_results.csv', index=False)

def parse_subchirp_file(file: str) -> (str, int, float, float, float, float):
    from MQTT_AAC_subchirp_test import get_cycles

    print(file)

    if "\\fixed\\" in file or "/fixed/" in file:
        conf: str = "fixed"
    else:
        conf: str = "dynamic"

    offset: int = int(file.split("_")[-4][-1])
    Ts: float = float(file.split("_")[-3])
    cycles_: float = float(file.split("_")[-2])  # Seems to be incorrect
    cycles: float = get_cycles({"configuration": conf, "symbol_time": Ts, "fstart": 9500, "fend": 13500})

    encoder = OChirpEncode(T=None, fs=9500, fe=13500, T_preamble=0, orthogonal_pair_offset=offset,
                           minimize_sub_chirp_duration=conf == "dynamic",
                           required_number_of_cycles=cycles)

    print(f"{cycles_} == {cycles}")
    print(f"{Ts} == {encoder.T}")

    decoder = OChirpDecode(encoder=encoder, original_data="UUUU")
    decoder.preamble_min_peak = 1000

    ber = decoder.decode_file(file, plot=False)

    # conf, offset, cycles, ber
    return conf, offset, Ts, cycles, ber


def parse_subchirp_test():
    dir = './data/results/06-01-2022-dynamic-vs-fixed-sub-chirps/Recorded_files/'
    # dir = 'E:/Recorded_files/'
    files = glob(dir + '**/*.wav', recursive=True)

    results = []
    with Pool(10) as p:
        results.extend(p.map(parse_subchirp_file, files))

    df = pd.DataFrame(results, columns=["configuration", "offset", "Ts", "cycles", "ber"])
    print(df)
    df.to_csv(dir + 'parsed_results.csv', index=False)

if __name__ == '__main__':
    # parse(n_extra_transmitters=3)

    parse_subchirp_test()
