from matplotlib import pyplot as plt
from OChirpEncode import OChirpEncode
from OChirpDecode import OChirpDecode
import numpy as np
import pandas as pd
from configuration import Configuration, get_configuration_encoder


"""
    This is file is meant to produce figures for the report in a consistent way
"""
fontSize = 14


def plot_example_ochirp():
    encoder = OChirpEncode(fsample=4 * 44100, T=None, fs=500, fe=2500, minimize_sub_chirp_duration=True, required_number_of_cycles=3)
    ochirp1 = encoder.get_single_chirp(0)
    encoder.minimal_sub_chirp_duration = False
    ochirp2 = encoder.get_single_chirp(0)[:-2]
    print(ochirp1.size)
    print(ochirp2.size)

    t = np.linspace(0, encoder.T*1000, len(ochirp1))

    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    axs[0].plot(t, ochirp2)
    axs[0].set_xlabel("a)")
    axs[1].plot(t, ochirp1)
    axs[1].set_xlabel("b)\nTime [ms]")
    axs[0].set_ylabel("Amplitude")
    axs[1].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("./images/example_ochirp.pdf", format="pdf", bbox_inches='tight')
    np.savetxt("./images/example_ochirp_min.csv", ochirp1, delimiter=",")
    np.savetxt("./images/example_ochirp_regular.csv", ochirp2, delimiter=",")
    plt.show()


def plot_example_frame():
    data = "H"
    encoder = get_configuration_encoder(Configuration.balanced)
    encoder.fsample = 4 * 44100
    _, frame = encoder.convert_data_to_sound(data, filename="temp.wav")
    frame = frame/np.max(frame)

    t = np.linspace(0, len(frame)/encoder.fsample*1000, len(frame))

    plt.figure(figsize=(6, 2))
    plt.xlim(-1, 120)
    plt.plot(t, frame)
    plt.fill_between([0, 48], -1.1, 1.1, alpha=0.5, color='green')
    plt.fill_between([48.1, 120], -1.1, 1.1, alpha=0.5, color='red')
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("./images/example_frame.pdf", format="pdf", bbox_inches='tight')
    np.savetxt("./images/exaple_frame.csv", frame, delimiter=",")
    plt.show()


def plot_example_decode():
    data = "H"
    encoder = get_configuration_encoder(Configuration.balanced)
    encoder.fsample = 4 * 44100
    _, frame = encoder.convert_data_to_sound(data, filename="temp.wav")
    frame = frame/np.max(frame)

    decoder = OChirpDecode(original_data=data, encoder=encoder)

    decoder.decode_data_raw(data=frame, plot=True)

    plt.gcf().suptitle("")
    plt.tight_layout()
    plt.savefig("./images/example_decode.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_example_peak_detection():
    data = "H"
    encoder = get_configuration_encoder(Configuration.balanced)
    encoder.fsample = 4 * 44100
    _, frame = encoder.convert_data_to_sound(data, filename="temp.wav")
    frame = frame / np.max(frame)

    decoder = OChirpDecode(original_data=data, encoder=encoder)

    decoder.decode_data_raw(data=frame, plot=True)

    plt.figure(2).suptitle("")
    plt.figure(2).set_size_inches(6, 3)
    plt.xlim(encoder.T_preamble * 1000, 1000 * (encoder.T_preamble + 3 * encoder.T))
    plt.tight_layout()
    plt.savefig("./images/example_peak_detection.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_range_test_results():
    # df = pd.read_csv("./data/results/30-11-2021/raw_test_results.csv")
    df = pd.read_csv('./data/results/31-12-2021-multi-transmitter-los/parsed_results.csv')

    color_list = ["#7e1e9c", '#0343df', '#43a2ca', '#0868ac', '#eff3ff', '#0000ff']

    config_list = ["Baseline (T=48ms)", "Baseline (T=24ms)", "Advanced (T~=24ms)", "Speed (T~=?)"]
    print(df.Configuration.unique())
    plt.figure(figsize=(6, 3))
    index = 0
    labels = []
    for distance in df.distance.unique():
        for i, config in enumerate(Configuration):
            print(f"{distance} {config}")
            data = df[(df.Configuration == str(config)) & (df.distance == distance)]

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

            labels.append(f"{config.name}")
            index += 1
            plt.axvline(x=index - 0.5, color='black', alpha=0.2)

        if distance < df.distance.max():
            plt.axvline(x=index - 0.5, color='r', linestyle="dashed")

    # plt.ylim(-0.025, 0.7)
    # plt.xlim(-0.5, 5.5)
    plt.text(x=9 - 0.5, y=0.8, s=f"Distance [m]", color='black', horizontalalignment='center',
             verticalalignment='center')
    plt.ylabel("BER")
    plt.xlabel("Configurations")
    plt.xticks(rotation=45)
    # labels.append("")
    plt.xticks(np.arange(index), labels)
    plt.tight_layout()
    plt.savefig("./images/range_test_results.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_multi_transmitter_range_test_results():
    # df1 = pd.read_csv('./data/results/05-01-2022-multi-transmitter-reverberend/parsed_results_250.csv')
    df2 = pd.read_csv('./data/results/07-01-2022-multi-transmitter-los/parsed_results_200cm.csv')
    # df3 = pd.read_csv('./data/results/05-01-2022-multi-transmitter-reverberend/parsed_results_150.csv')
    df4 = pd.read_csv('./data/results/07-01-2022-multi-transmitter-los/parsed_results_100_150cm.csv')
    df5 = pd.read_csv('./data/results/07-01-2022-multi-transmitter-los/parsed_results_50cm.csv')

    df = pd.concat([df2, df4, df5])
    # df = pd.read_csv('./data/results/07-01-2022-multi-transmitter-los/parsed_results.csv')
    df.to_csv("./all_data.csv", index=False)

    color_list = ["#7e1e9c", '#0343df', '#43a2ca', '#0868ac', '#eff3ff', '#0000ff']
    configurations = ['Configuration.baseline', 'Configuration.optimized', 'Configuration.baseline48']

    df = df.sort_values(by=["distance", "transmitters"])

    print(df.Configuration.unique())
    print(df.distance.unique())

    plt.figure(figsize=(6, 3))
    index = 0
    labels = []
    for distance in df.distance.unique():
        for transmitters in df.transmitters.unique():
            for i, config in enumerate(configurations):
                data = df[(df.Configuration == str(config)) & (df.distance == distance) & (df.transmitters == transmitters)]

                print(f"{distance} {transmitters} {config} {np.mean(data.ber)}")

                medianprops = {'color': color_list[i], 'linewidth': 2}
                boxprops = {'color': color_list[i], 'linestyle': '-'}
                whiskerprops = {'color': color_list[i], 'linestyle': '-'}
                capprops = {'color': color_list[i], 'linestyle': '-'}

                plt.boxplot(data['ber'], positions=[-0.33 + index + i*0.33], showfliers=False, medianprops=medianprops, boxprops=boxprops,
                            whiskerprops=whiskerprops, capprops=capprops, widths=0.65/3)

            # hardcoded to be at the middle on the x-axis
            if int(index + len(df.transmitters.unique())/2) % len(df.transmitters.unique()) == 0:
                plt.text(x=index - 0.5, y=0.82, s=f"{distance}", color='black', horizontalalignment='center',
                         verticalalignment='center')

            labels.append(f"{transmitters}")
            index += 1
            plt.axvline(x=index - 0.5, color='black', alpha=0.2)

        if distance < df.distance.max():
            plt.axvline(x=index - 0.5, color='r', linestyle="dashed")

    plt.ylim(-0.025, 0.8)
    plt.xlim(-0.5, index - 0.5)
    plt.text(x=index/2 - 0.5, y=0.9, s=f"Distance [cm]", color='black', horizontalalignment='center',
             verticalalignment='center')
    plt.ylabel("BER")
    plt.xlabel("Transmitters")
    plt.xticks(np.arange(index), labels, rotation=0, ha='center')

    for i, _ in enumerate(df.Configuration.unique()):
        plt.scatter(0, -1, color=color_list[i], marker=None, label=configurations[i].split('.')[-1])
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("./images/range_test_results_mt.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_effective_bit_rate():
    df1 = pd.read_csv('./data/results/05-01-2022-multi-transmitter-nlos/parsed_results_250.csv')
    df2 = pd.read_csv('./data/results/05-01-2022-multi-transmitter-nlos/parsed_results_200.csv')
    df3 = pd.read_csv('./data/results/05-01-2022-multi-transmitter-nlos/parsed_results_150.csv')
    df4 = pd.read_csv('./data/results/05-01-2022-multi-transmitter-nlos/parsed_results_100.csv')
    df5 = pd.read_csv('./data/results/05-01-2022-multi-transmitter-nlos/parsed_results_50.csv')

    df = pd.concat([df1, df2, df3, df4, df5])

    color_list = ["#7e1e9c", '#0343df', '#43a2ca', '#0868ac', '#eff3ff', '#0000ff']
    configurations = ['Configuration.baseline', 'Configuration.optimized']
    bitrates = {'Configuration.baseline': 41.67,
                'Configuration.optimized': 53.2
                }
    markers = ['+', 'D']

    df = df.sort_values(by=["distance", "transmitters"])

    print(df.Configuration.unique())
    print(df.distance.unique())

    df = df.groupby(["distance", "transmitters", "Configuration"]).ber.agg(["mean", "std"]).reset_index()
    df['bitrate'] = df.Configuration
    df = df.replace({"bitrate": bitrates})
    df['eff_bitrate'] = (1 - df["mean"]) * df.bitrate

    # print(df)

    plt.figure(figsize=(6, 3))
    for transmitters in df.transmitters.unique():
        for i, config in enumerate(configurations):
            data = df[(df.Configuration == str(config)) & (df.transmitters == transmitters)]

            print(f"{transmitters} {config} {np.mean(data['mean'])}")
            plt.plot(data.distance, data.eff_bitrate, label=f'{config}-{transmitters}', marker=markers[i])

    plt.legend()
    plt.savefig("./images/effective_bit_rates.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_subchirp_test():
    df = pd.read_csv('./data/results/06-01-2022-dynamic-vs-fixed-sub-chirps/parsed_results.csv')

    configurations = ["dynamic", "fixed"]

    df = df.groupby(["configuration", "Ts"]).ber.agg(["std", "mean"]).reset_index()

    print(df)

    plt.figure(figsize=(6, 3))
    for i, conf in enumerate(configurations):
        data = df[df.configuration == conf]

        plt.errorbar(data.Ts * 1000, data["mean"], label=conf)

    # plt.hlines(0.01, 0, 25, linestyles='dotted', color='black')
    plt.ylim(-0.00001, 1.1)
    plt.ylabel("BER")
    plt.xlabel("Symbol Time [ms]")
    plt.grid()
    plt.yscale("symlog", linthresh=0.0001)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # plot_example_ochirp()
    # plot_example_frame()
    # plot_example_decode()
    # plot_example_peak_detection()
    # plot_range_test_results()
    plot_multi_transmitter_range_test_results()
    # plot_effective_bit_rate()
    # plot_subchirp_test()


if __name__ == "__main__":
    main()
