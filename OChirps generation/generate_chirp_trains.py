from AAC.OChirpEncode import OChirpEncode
from scipy.io.wavfile import write, read
import numpy as np


def generate_chirp_train(chirp: str, N: int, silence_time: float, preamble_time: float, preamble_range: (int, int)):
    """
        Generate a chirp train based on the given parameters
    """
    encoder = OChirpEncode(T_preamble=preamble_time,
                           f_preamble_start=preamble_range[0],
                           f_preamble_end=preamble_range[1])

    preamble = encoder.get_preamble()
    fs, data = read(chirp)

    if fs != encoder.fsample:
        print("Encoder sample rate not the same as the chirp files!\n Re-generate the chirp files or fix the encoder.")

    silence = np.zeros(int(silence_time * encoder.fsample))

    signal = np.append(preamble, silence)
    for _ in range(N):
        signal = np.append(signal, np.append(data, silence))

    write(f"chirp_train_{chirp}", encoder.fsample, signal.astype(np.float32))


def generate_chirp_files():
    """
        Generate the chirp .wav files. Should not have to be done often.
    """
    encoder = OChirpEncode(T=0.024)

    for i in range(encoder.M):
        print(f"Creating chirp: {i}")
        chirp_data = encoder.get_single_chirp(i)
        write(f"chirp{i}.wav", encoder.fsample, chirp_data)


if __name__ == "__main__":
    generate_chirp_files()
    for i in range(8):
        print(f"Generating chirp train for chirp: {i}")
        generate_chirp_train(f"chirp{i}.wav", N=200, silence_time=0.1, preamble_time=0.25, preamble_range=(1, 5500))
