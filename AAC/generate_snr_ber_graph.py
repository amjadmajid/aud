from scipy.io.wavfile import read, write
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import sounddevice as sd
from multiprocessing import Pool
from configuration import Configuration, get_configuration_encoder
from OChirpDecode import OChirpDecode
import pandas as pd
from pathlib import Path

# Noise source: https://freesound.org/people/InspectorJ/sounds/403180/


def get_mean_signal_power(data: np.ndarray):
    return np.mean(data ** 2)


def get_mean_signal_power_db(data: np.ndarray):
    return 10 * np.log10(get_mean_signal_power(data))


def normalize(data: np.ndarray) -> np.ndarray:
    return ((data/np.max(data)) * np.iinfo(np.int16).max).astype(np.int16)


def get_desired_data_gain(data: np.ndarray, noise: np.ndarray, desired_snr_db: float):
    noise_power = get_mean_signal_power_db(noise)
    data_power = get_mean_signal_power_db(data)

    current_snr_db = data_power - noise_power
    # print(f"Original: {current_snr_db:.2f} dB = {data_power:.2f} - {noise_power:.2f}")

    snr_adjustment = current_snr_db - desired_snr_db
    data_gain = 10 ** (snr_adjustment/20)
    return data_gain


def add_noise_to_data(data: np.ndarray, noise: np.ndarray, desired_snr_db: float):
    data_gain = get_desired_data_gain(data, noise, desired_snr_db)

    adjusted_data = data / data_gain
    # adjusted_data_power = get_mean_signal_power_db(adjusted_data)

    # current_snr_db = adjusted_data_power - noise_power
    # print(f"Adjusted: {current_snr_db:.2f} = {adjusted_data_power:.2f} / {noise_power:.2f}")

    return adjusted_data + noise


def generate_noisy_samples():
    babble_noise_file = "./babble_noise.wav"
    pure_signals = glob("./sample_chirps/*.wav")
    snrs = np.arange(-50, 1, 5)
    num_iterations = 30

    fs, bubble_noise = read(babble_noise_file)
    bubble_noise = bubble_noise.astype(np.float64)

    for i, signal in enumerate(pure_signals):
        fs_, data = read(signal)
        data = data.astype(np.float64)

        if fs != fs_:
            print("Error! Sample speeds are not the same!")
            exit()

        for snr in snrs:
            for iteration in range(num_iterations):
                print(snr)

                data_gain = get_desired_data_gain(data, bubble_noise, snr)
                scaled_data = data / data_gain

                # Select a random section of noise
                # And add a second of the other noise at the beginning and end to make the clip more realistic
                random_offset = np.random.randint(data.size, bubble_noise.size - data.size)

                begin = random_offset - fs
                if begin < 0:
                    begin = 0
                end = random_offset + data.size + fs
                if end > bubble_noise.size:
                    end = bubble_noise.size
                random_noise = np.array(bubble_noise[begin:end], copy=True)

                random_noise[fs:fs + scaled_data.size] += scaled_data
                noised_data = random_noise

                noised_data = noised_data.astype(np.int16)

                config = signal.split('\\')[-1].replace('.wav', '')

                dir = f'./sample_chirps/noised/{config}/'
                Path(dir).mkdir(parents=True, exist_ok=True)
                write(filename=dir + f"{config}_{iteration}_{snr}dB.wav", data=noised_data, rate=fs)


def decode_noisy_sample(sample: str) -> (Configuration, int, float):
    config_name = sample.split("/")[-1].split("\\")[1]
    offset = int(config_name[-1])
    config = Configuration[config_name[:-1]]
    snr = float(sample.split("_")[-1].replace("dB.wav", ""))
    iteration = int(sample.split("_")[-2])

    if iteration < 300:
        encoder = get_configuration_encoder(config)
        encoder.orthogonal_pair_offset = offset
        decoder = OChirpDecode(encoder=encoder, original_data="Help")
        return config, snr, decoder.decode_file(sample, plot=False)
    else:
        return None, None, None


def process_noisy_samples():
    noisy_samples = glob('./sample_chirps/noised/**/*.wav', recursive=True)

    # decode_noisy_sample(noisy_samples[0])

    with Pool(10) as p:
        results = p.map(decode_noisy_sample, noisy_samples)

    df = pd.DataFrame(results, columns=['Configuration', 'snr', 'ber'])
    df = df.dropna()
    df.to_csv("./data/results/snr_ber_data.csv", index=False)


def plot_noisy_samples():
    df = pd.read_csv("./data/results/snr_ber_data.csv")
    configurations = df.Configuration.unique()

    markers = ['+', 'x', 'd', 'D']

    df = df.groupby(['Configuration', 'snr']).ber.agg(['mean', 'std']).reset_index()
    print(configurations)

    plt.figure()
    plt.ylabel("BER")
    plt.xlabel("SNR [dB]")
    plt.yscale('symlog', linthresh=10**-5)
    plt.ylim(-0.00000075, 1.1)
    plt.grid(True)
    # print(df.snr.unique())
    for i, conf in enumerate(configurations):
        data = df[df.Configuration == conf]

        plt.plot(data.snr, data['mean'], label=conf.split(".")[-1], marker=markers[i])
        # plt.fill_between(data.snr, data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.33)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # generate_noisy_samples()
    # process_noisy_samples()
    plot_noisy_samples()



