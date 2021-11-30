from matplotlib import pyplot as plt
from OChirpEncode import OChirpEncode
from OChirpDecode import OChirpDecode
import numpy as np
import pandas as pd


"""
    This is file is meant to produce figures for the report in a consistent way
"""


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


def plot_range_test_results():
    df = pd.read_csv("./data/results/30-11-2021/raw_test_results.csv")

    color_list = ["#7e1e9c", '#0343df', '#43a2ca', '#0868ac', '#eff3ff', '#0000ff']

    config_list = ["Baseline (T=48ms)", "Baseline (T=24ms)", "Advanced (T~=24ms)", "Speed (T~=?)"]

    plt.figure(figsize=(6, 3))
    index = 0
    labels = []
    for distance in df.distance.unique():
        for i, config in enumerate(df.Configuration.unique()):
            print(f"{distance} {config}")
            data = df[(df.Configuration == config) & (df.distance == distance)]

            medianprops = {'color': color_list[i], 'linewidth': 2}
            boxprops = {'color': color_list[i], 'linestyle': '-'}
            whiskerprops = {'color': color_list[i], 'linestyle': '-'}
            capprops = {'color': color_list[i], 'linestyle': '-'}

            plt.boxplot(data['ber'], positions=[index], showfliers=False, medianprops=medianprops, boxprops=boxprops,
                        whiskerprops=whiskerprops, capprops=capprops, widths=0.65)
            # hardcoded to be at the middle on the x-axis
            if int(index + len(df.Configuration.unique())/2) % len(df.Configuration.unique()) == 0:
                plt.text(x=index-0.5, y=0.725, s=f"{distance}", color='black', horizontalalignment='center',
                         verticalalignment='center')

            labels.append(f"{config}")
            index += 1
            plt.axvline(x=index - 0.5, color='black', alpha=0.2)

        if distance < df.distance.max():
            plt.axvline(x=index - 0.5, color='r', linestyle="dashed")

    plt.ylim(-0.025, 0.7)
    plt.xlim(-0.5, 11.5)
    plt.text(x=9 - 0.5, y=0.8, s=f"Distance [m]", color='black', horizontalalignment='center',
             verticalalignment='center')
    plt.ylabel("BER")
    plt.xlabel("Configurations")
    plt.xticks(np.arange(index), labels)
    plt.tight_layout()
    plt.savefig("./images/range_test_results.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def main():
    # plot_example_ochirp()
    # plot_example_frame()
    # plot_example_decode()
    # plot_example_peak_detection()
    plot_range_test_results()


if __name__ == "__main__":
    main()
