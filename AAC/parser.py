from glob import glob
import os
from configuration import Configuration, get_configuration_encoder
from OChirpDecode import OChirpDecode
import pandas as pd

directory = './data/results/15-12-2021/'
configurations = ['baseline', 'baseline_fast', 'balanced', 'fast']

if __name__ == '__main__':
    files = glob(directory + '**/*.wav', recursive=True)

    bers = []

    for file in files:

        for conf in configurations:
            if '\\' + conf + '\\' in file:
                break

        conf = Configuration[conf]
        encoder = get_configuration_encoder(conf)
        decoder = OChirpDecode(encoder=encoder, original_data="Hello, World!")
        filename = os.path.basename(file)
        distance = int(filename.split('_')[1].replace('cm', ''))

        ber = decoder.decode_file(file, plot=False)
        bers.append((conf, distance, ber))
        # if ber != 0.0:
        #     decoder.decode_file(file, plot=True)

    df = pd.DataFrame(bers, columns=['Configuration', 'distance', 'ber'])

    df.to_csv(directory + 'parsed_results.csv', index=False)
