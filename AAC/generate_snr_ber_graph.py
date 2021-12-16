from scipy.io.wavfile import read, write
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import sounddevice as sd
from multiprocessing import Pool
from configuration import Configuration, get_configuration_encoder
from OChirpDecode import OChirpDecode
import pandas as pd


# Noise source: https://freesound.org/people/InspectorJ/sounds/403180/


def get_mean_signal_power(data: np.ndarray):
    return np.mean(data ** 2)


def get_mean_signal_power_db(data: np.ndarray):
    return 10 * np.log10(get_mean_signal_power(data))


def normalize(data: np.ndarray) -> np.ndarray:
    return data/np.max(data)


def add_noise_to_data(data: np.ndarray, noise: np.ndarray, desired_snr_db: float):
    noise_power = get_mean_signal_power_db(noise)
    data_power = get_mean_signal_power_db(data)

    current_snr_db = data_power - noise_power
    # print(f"Original: {current_snr_db:.2f} dB = {data_power:.2f} - {noise_power:.2f}")

    snr_adjustment = current_snr_db - desired_snr_db
    data_gain = 10 ** (snr_adjustment/20)
    # print(f"Calculated data gain {data_gain:.3f}")

    adjusted_data = data / data_gain
    # adjusted_data_power = get_mean_signal_power_db(adjusted_data)

    # current_snr_db = adjusted_data_power - noise_power
    # print(f"Adjusted: {current_snr_db:.2f} = {adjusted_data_power:.2f} / {noise_power:.2f}")

    return adjusted_data + noise


def generate_noisy_samples():
    babble_noise_file = "./babble_noise.wav"
    pure_signals = glob("./sample_chirps/*.wav")
    snrs = np.arange(-50, 0, 1)
    num_iterations = 10

    fs, bubble_noise = read(babble_noise_file)
    bubble_noise = bubble_noise.astype(np.float64)

    for i, signal in enumerate(pure_signals):
        fs_, data = read(signal)

        if fs != fs_:
            print("Error! Sample speeds are not the same!")
            exit()

        for snr in snrs:
            for iteration in range(num_iterations):
                print(snr)
                random_offset = np.random.randint(data.size, bubble_noise.size - data.size)
                signal_with_noise = add_noise_to_data(data.astype(np.float64), bubble_noise[random_offset:random_offset+data.size], desired_snr_db=snr)
                signal_with_noise = normalize(signal_with_noise)

                config = signal.split('\\')[-1].replace('.wav', '')
                write(filename=f'./sample_chirps/noised/{config}/{config}_{iteration}_{snr}dB.wav', data=signal_with_noise, rate=fs)


def decode_noisy_sample(sample: str) -> (Configuration, int, float):
    config_name = sample.split("/")[-1].split("\\")[1]
    config = Configuration[config_name]
    snr = int(sample.split("_")[-1].replace("dB.wav", ""))

    encoder = get_configuration_encoder(config)
    decoder = OChirpDecode(encoder=encoder, original_data="Hello, World!")
    return config, snr, decoder.decode_file(sample, plot=False)


def process_noisy_samples():
    noisy_samples = glob('./sample_chirps/noised/**/*.wav', recursive=True)

    # decode_noisy_sample(noisy_samples[0])

    with Pool(12) as p:
        results = p.map(decode_noisy_sample, noisy_samples)

    df = pd.DataFrame(results, columns=['Configuration', 'snr', 'ber'])
    df.to_csv("./data/results/snr_ber_data.csv", index=False)


def plot_noisy_samples():
    df = pd.read_csv("./data/results/snr_ber_data.csv")
    configurations = df.Configuration.unique()

    df = df.groupby(['Configuration', 'snr']).ber.agg(['mean', 'std']).reset_index()
    print(configurations)

    for conf in configurations:
        plt.figure()
        data = df[df.Configuration == conf]

        plt.plot(data.snr, data['mean'], label=conf)

        plt.legend()
    plt.show()


if __name__ == '__main__':
    # generate_noisy_samples()
    process_noisy_samples()
    plot_noisy_samples()



