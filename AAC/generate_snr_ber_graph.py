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
from itertools import repeat

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


def generate_noisy_sample(pure_signal: np.ndarray, noise_data: np.ndarray, snr: float, fs: int = 44100):
    print(snr)

    data_gain = get_desired_data_gain(pure_signal, noise_data, snr)
    scaled_data = pure_signal / data_gain

    # Select a random section of noise
    # And add a second of the other noise at the beginning and end to make the clip more realistic
    random_offset = np.random.randint(pure_signal.size, noise_data.size - pure_signal.size)

    begin = random_offset - fs
    if begin < 0:
        begin = 0
    end = random_offset + pure_signal.size + fs
    if end > noise_data.size:
        end = noise_data.size
    random_noise = np.array(noise_data[begin:end], copy=True)

    random_noise[fs:fs + scaled_data.size] += scaled_data
    noised_data = random_noise

    noised_data = noised_data.astype(np.int16)

    return noised_data


def generate_noisy_samples(sample_location: str = "./sample_chirps/", num_iterations: int = 30):
    babble_noise_file = "./babble_noise.wav"
    pure_signals = glob(sample_location + '*.wav')
    snrs = np.arange(-50, 1, 5)

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

                noised_data = generate_noisy_sample(data, bubble_noise, snr)

                config = signal.split('\\')[-1].replace('.wav', '')

                dir = f'{sample_location}/noised/{config}/'
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


def process_noisy_samples(noised_samples: str = "./sample_chirps/noised/"):
    noisy_samples = glob(f'.{noised_samples}**/*.wav', recursive=True)

    # decode_noisy_sample(noisy_samples[0])

    with Pool(10) as p:
        results = p.map(decode_noisy_sample, noisy_samples)

    df = pd.DataFrame(results, columns=['Configuration', 'snr', 'ber'])
    df = df.dropna()
    df.to_csv("./data/results/snr_ber_data.csv", index=False)


def plot_noisy_samples(data: str = "./data/results/snr_ber_data.csv"):
    df = pd.read_csv(data)
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


def main():
    # generate_noisy_samples()
    # process_noisy_samples()
    plot_noisy_samples()


def generate_effective_bit_rate_per_snr(file_to_save: str = "./effective_bitrate_snr_simulation_results.csv"):
    from OChirpEncode import OChirpEncode

    results = []

    configurations = ["baseline", "optimized"]
    offsets = [0, 2, 4, 6]

    cycles_to_test = range(1, 21, 5)
    extra_transmitters = range(0, 4)
    repeats = range(2)

    for L in cycles_to_test:
        # Generate all configurations for this L
        encoders = {
            "baseline": {
                0: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=0,
                                minimize_sub_chirp_duration=False, required_number_of_cycles=L),
                2: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=2,
                                minimize_sub_chirp_duration=False, required_number_of_cycles=L),
                4: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=4,
                                minimize_sub_chirp_duration=False, required_number_of_cycles=L),
                6: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=6,
                                minimize_sub_chirp_duration=False, required_number_of_cycles=L)
                },
            "optimized": {
                0: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=0,
                                minimize_sub_chirp_duration=True, required_number_of_cycles=L),
                2: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=2,
                                minimize_sub_chirp_duration=True, required_number_of_cycles=L),
                4: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=4,
                                minimize_sub_chirp_duration=True, required_number_of_cycles=L),
                6: OChirpEncode(T=None, T_preamble=0, orthogonal_pair_offset=6,
                                minimize_sub_chirp_duration=True, required_number_of_cycles=L)
            }
        }
        datas = {
            "baseline": {
                0: encoders["baseline"][0].convert_data_to_sound(data="UUUU", filename=None)[-1],
                2: encoders["baseline"][2].convert_data_to_sound(data="UUUU", filename=None)[-1],
                4: encoders["baseline"][4].convert_data_to_sound(data="UUUU", filename=None)[-1],
                6: encoders["baseline"][6].convert_data_to_sound(data="UUUU", filename=None)[-1],
                },
            "optimized": {
                0: encoders["optimized"][0].convert_data_to_sound(data="UUUU", filename=None)[-1],
                2: encoders["optimized"][2].convert_data_to_sound(data="UUUU", filename=None)[-1],
                4: encoders["optimized"][4].convert_data_to_sound(data="UUUU", filename=None)[-1],
                6: encoders["optimized"][6].convert_data_to_sound(data="UUUU", filename=None)[-1],
            }
        }
        decoders = {
            "baseline": {
                0: OChirpDecode(encoder=encoders["baseline"][0], original_data="UUUU"),
                2: OChirpDecode(encoder=encoders["baseline"][2], original_data="UUUU"),
                4: OChirpDecode(encoder=encoders["baseline"][4], original_data="UUUU"),
                6: OChirpDecode(encoder=encoders["baseline"][6], original_data="UUUU"),
                },
            "optimized": {
                0: OChirpDecode(encoder=encoders["optimized"][0], original_data="UUUU"),
                2: OChirpDecode(encoder=encoders["optimized"][2], original_data="UUUU"),
                4: OChirpDecode(encoder=encoders["optimized"][4], original_data="UUUU"),
                6: OChirpDecode(encoder=encoders["optimized"][6], original_data="UUUU"),
            }
        }

        def run_iteration(conf, offset, n):
            encoder = encoders[conf][offset]
            data = datas[conf][offset]
            decoder = decoders[conf][offset]

            other_offsets = offsets.copy()
            other_offsets.remove(offset)
            np.random.shuffle(other_offsets)

            for i in range(n):
                noise_data = datas[conf][2 * i]
                start = np.random.randint(0, data.size // 2)
                end = noise_data.size
                data[start:end] += noise_data[:end - start]

            ber = decoder.decode_data(data, plot=False)

            return L, conf, 1/encoder.T, offset, n, ber

        # def run_iteration_async(setting):
        #     return run_iteration(setting[0], setting[1], setting[2])

        settings = []
        for conf in configurations:
            for offset in offsets:
                for n in extra_transmitters:
                    for r in repeats:
                        settings.append((conf, offset, n, r))

        # with Pool(9) as p:
        #     results = p.map(run_iteration_async, settings)

        for conf in configurations:
            for offset in offsets:
                for n in extra_transmitters:
                    for _ in repeats:
                        res = run_iteration(conf, offset, n)
                        results.append(res)

    df = pd.DataFrame(results, columns=["L", "configuration", "bitrate", "offset", "transmitters", "ber"])
    df.to_csv(file_to_save, index=False)
    print(df)


def plot_effective_bit_rate_per_snr(file_to_read: str = "./effective_bitrate_snr_simulation_results.csv"):
    df = pd.read_csv(file_to_read)

    df = df.groupby(["L", "configuration", "bitrate", "transmitters"]).ber.agg(["mean", "std"]).reset_index()
    df['eff_bitrate'] = (1 - df["mean"]) * df.bitrate

    df = df.sort_values(by="L")

    configurations = ["baseline", "optimized"]
    markers = ['+', "D"]
    transmitters = range(0, 4)

    plt.figure(figsize=(6, 3))
    for i, conf in enumerate(configurations):
        for transmitter in transmitters:
            data = df[(df.configuration == conf) & (df.transmitters == transmitter)]
            print(data)
            plt.plot(data.L, data.eff_bitrate, label=conf + "-" + str(transmitter), marker=markers[i])

    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./images/effective_bit_rate_over_L.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    # main()
    generate_effective_bit_rate_per_snr()
    plot_effective_bit_rate_per_snr()


