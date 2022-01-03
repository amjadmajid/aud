from glob import glob
from configuration import Configuration, get_configuration_encoder
from OChirpDecode import OChirpDecode
import pandas as pd
import os
import numpy as np
from scipy.io.wavfile import read

directory = './data/results/03-01-2022-multi-transmitter-los/'
configurations = ['baseline', 'halved_cycles', 'increased_freq', 'dynamic_subchirp']
chirp_pair_offsets = [0, 2, 4, 6]

all_configs = []
for c in configurations:
    for o in chirp_pair_offsets:
        all_configs.append(c + str(o))


def calculate_ber_multi_transmitter(file, number_of_transmitters, possible_files, decoder):
    print(f"{number_of_transmitters} transmitters")

    random_files_selection = possible_files.copy()

    # Select number_of_transmitters random files from this array (no redraws)
    selected_files = []
    for _ in range(number_of_transmitters):
        selected_files.append(random_files_selection[np.random.randint(0, len(random_files_selection))])
        random_files_selection.remove(selected_files[-1])

    # Read data from extra transmitters and superimpose it on the original data
    _, original_data = read(file)
    for selected_file in selected_files:
        _, data = read(selected_file)
        offset = np.random.randint(0, original_data.size) - (original_data.size // 2)
        if offset < 0:
            offset = 0
        original_data[offset:] = original_data[offset:] + data[offset:]

    # Get the channel with the highest energy
    print(original_data.shape)
    if len(original_data.shape) > 1 and original_data.shape[1] > 1:
        original_data = original_data[:, np.argmax(np.max(original_data, axis=0), axis=0)]
    print(original_data.shape)

    # decode the data
    ber = decoder.decode_data(original_data, plot=False)
    # if ber > 0:
    #     decoder.decode_data(original_data, plot=True)
    return ber


def parse(n_extra_transmitters: int = 0):
    files = glob(directory + '**/*.wav', recursive=True)

    # We randomly superimpose other transmitters on this one, so we need to make sure that we randomize it a bit
    # So this is how often we repeat (randomly) every file/transmitter combination
    extra_transmitters_iterations = 10

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

        if distance != 250:
            continue

        if n_extra_transmitters == 0:
            ber = decoder.decode_file(file, plot=False)
            bers.append((conf, distance, 1, ber))
        else:
            ber = decoder.decode_file(file, plot=False)
            bers.append((conf, distance, 1, ber))
            for extra_transmitters in range(1, n_extra_transmitters + 1):
                # Get possible configs that may operate as noise
                possible_configs = all_configs.copy()
                print(config_name)
                possible_configs.remove(config_name)

                # Get all valid noise files
                random_files_selection = []
                for possible_config in possible_configs:
                    if conf.name in possible_config and (('_' in conf.name) == ('_' in possible_config)):
                        new_file_path = file_path.replace(config_name, possible_config)
                        random_files_selection.extend(glob(new_file_path + '\\*.wav', recursive=False))

                # Make sure we average the randomness of the noise
                for _ in range(extra_transmitters_iterations):
                    ber = calculate_ber_multi_transmitter(file, extra_transmitters, random_files_selection, decoder)
                    bers.append((conf, distance, extra_transmitters + 1, ber))

        # if ber > 0:
        #     print("\nINFORMATION")
        #     print(file)
        #     print(conf)
        #     print(f"offset: {offset}")
        #     print(f"{distance}cm")
        #     decoder.decode_file(file, plot=True)

    df = pd.DataFrame(bers, columns=['Configuration', 'distance', 'transmitters', 'ber'])

    df.to_csv(directory + 'parsed_results.csv', index=False)

    print(f"avg ber: {df.ber.mean()}")


if __name__ == '__main__':
    parse(n_extra_transmitters=3)
