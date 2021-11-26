from matplotlib import pyplot as plt
from OChirpEncode import OChirpEncode
from OChirpDecode import OChirpDecode
import numpy as np


def plot_example_ochirp():
    encoder = OChirpEncode(fsample=44100*4)
    ochirp = encoder.get_single_chirp(0)

    t = np.linspace(0, encoder.T*1000, len(ochirp))

    plt.figure(figsize=(6, 2))
    plt.plot(t, ochirp)
    plt.xlabel("Time [ms]")
    plt.tight_layout()
    plt.savefig("./images/example_ochirp.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_example_frame():
    data = "Hello"
    encoder = OChirpEncode(fsample=44100*4)
    _, frame = encoder.convert_data_to_sound(data, filename="temp.wav")

    t = np.linspace(0, len(frame)/encoder.fsample*1000, len(frame))

    plt.figure(figsize=(6, 2))
    plt.plot(t, frame)
    plt.xlabel("Time [ms]")
    plt.tight_layout()
    plt.savefig("./images/example_frame.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_example_decode():
    data = "Hello"
    encoder = OChirpEncode(fsample=44100*4)
    _, frame = encoder.convert_data_to_sound(data, filename="temp.wav")

    decoder = OChirpDecode(original_data=data, encoder=encoder)

    decoder.decode_data_raw(data=frame, plot=True)

    plt.gcf().suptitle("")
    plt.tight_layout()
    plt.savefig("./images/example_decode.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_example_peak_detection():
    data = "Hello"
    encoder = OChirpEncode(fsample=44100*4)
    _, frame = encoder.convert_data_to_sound(data, filename="temp.wav")

    decoder = OChirpDecode(original_data=data, encoder=encoder)

    decoder.decode_data_raw(data=frame, plot=True)

    plt.figure(2).suptitle("")
    plt.figure(2).set_size_inches(6, 3)
    plt.xlim(encoder.T_preamble*encoder.fsample, encoder.T_preamble*encoder.fsample + 3*encoder.T*encoder.fsample)
    plt.tight_layout()
    plt.savefig("./images/example_peak_detection.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def main():
    # plot_example_ochirp()
    # plot_example_frame()
    # plot_example_decode()
    plot_example_peak_detection()


if __name__ == "__main__":
    main()
