from scipy.signal import chirp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 100000


# The following two functions are from:
# https://stackoverflow.com/questions/10237926/convert-string-to-list-of-bits-and-viceversa
def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def frombits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


# Figure 3.a in the paper
def plot_up_chirps():
    M = 5
    fs = 10000
    fe = 20000
    T = 0.1
    Tb = T / M
    for m in range(M):
        fsm = fs + (m * (fe - fs)) / M
        fem = fsm + (fe - fs) / M

        t = np.arange(0, int(Tb * fs)) / fs
        y = np.linspace(fsm, fem, len(t))

        for m_ in range(M):
            plt.plot(t + (m_ * Tb), y)

    plt.xlim(0, T)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Frequency [Hz]")


# Figure 3.b in the paper
def plot_down_chirps():
    M = 5
    fs = 10000
    fe = 20000
    T = 0.1
    Tb = T / M
    for m in range(M):
        fsm = fs + (m * (fe - fs)) / M
        fem = fsm + (fe - fs) / M

        t = np.arange(0, int(Tb * fs)) / fs
        y = np.linspace(fem, fsm, len(t))

        for m_ in range(M):
            plt.plot(t + (m_ * Tb), y)

    plt.xlim(0, T)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Frequency [Hz]")


# Figure 3.c in the paper
def plot_hybrid_chirps():
    M = 5
    fs = 10000
    fe = 20000
    T = 0.1
    Tb = T / M
    for m in range(1, M + 1):
        fsm = fs + ((m - 1) * (fe - fs)) / M
        fem = fsm + (fe - fs) / M

        t = np.arange(0, int(Tb * fs)) / fs
        y = np.linspace(fsm, fem, len(t))
        y_not = np.linspace(fem, fsm, len(t))

        start = int(m % 2 == 0)
        for i in range(start, M + start):
            if i % 2 == 0:
                plt.plot(t + ((i - start) * Tb), y)
            else:
                plt.plot(t + ((i - start) * Tb), y_not)

    plt.xlim(0, T)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Frequency [Hz]")


# Figure 4a-f in the paper
def plot_unity_pulses():
    M = 5
    T = 0.1
    Tb = T / M
    # R = [[4, 6, 2, 1, 5, 3],
    #      [3, 4, 6, 2, 1, 5],
    #      [5, 3, 4, 6, 2, 1],
    #      [1, 5, 3, 4, 6, 2],
    #      [2, 1, 5, 3, 4, 6],
    #      [6, 2, 1, 5, 3, 4]]

    R = [[5, 1, 3, 4, 2],
         [3, 5, 2, 1, 4],
         [2, 3, 4, 5, 1],
         [1, 4, 5, 2, 3],
         [4, 2, 1, 3, 5]]

    # fig = plt.figure()
    for r in R:
        fig, axs = plt.subplots(M, sharex=True, sharey=True)
        for i, element in enumerate(r):
            # print(element)
            start = i * Tb
            end = start + Tb
            x = np.linspace(0, T, 1000)
            y = np.logical_and(x >= start, x <= end)
            axs[5-element].plot(x, y)
            axs[5-element].set_ylim(-0.1, 1.1)
        axs[-1].set_xlabel("time [s]")
        fig.tight_layout()

        # plt.show()


def plot_both():
    M = 5
    fs = 10000
    fe = 20000
    T = 0.1
    Tb = T / M
    R = [[5, 1, 3, 4, 2],
         [3, 5, 2, 1, 4],
         [2, 3, 4, 5, 1],
         [1, 4, 5, 2, 3],
         [4, 2, 1, 3, 5]]

    fig = plt.figure(figsize=(6, 6))
    subfigs = fig.subfigures(2, 1)
    axs_top = subfigs[1].subplots(M, sharex=True, sharey=True)
    axs_bottom = subfigs[0].subplots(1, sharex=True)
    row = R[0]
    for i, element in enumerate(row):
        start = i * Tb
        end = start + Tb
        x = np.linspace(0, T, 1000)
        y = np.logical_and(x >= start, x <= end)
        axs_top[5 - element].plot(x, y)
        axs_top[5 - element].set_ylim(-0.1, 1.1)

        axs_top[element - 1].set_ylabel(f"{6-element}", rotation=0)
        axs_top[element - 1].yaxis.set_label_coords(-0.06, 0.275)
        if element == 3:
            axs_top[element - 1].text(-0.017, 0.5, "selected sub-chirp", rotation=90, in_layout=False, 
                                      horizontalalignment='center', verticalalignment='center')

        m = element
        fsm = fs + ((m - 1) * (fe - fs)) / M
        fem = fsm + (fe - fs) / M

        t = np.arange(0, int(Tb * fs)) / fs
        if m % 2 == i % 2:
            y = np.linspace(fem, fsm, len(t))
        else:
            y = np.linspace(fsm, fem, len(t))
        axs_bottom.plot(t + (i * Tb), y)
        axs_bottom.grid(True)
        axs_bottom.set_ylabel("frequency [Hz]")

    axs_top[-1].set_xlabel("time [s]")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.13)
    plt.show()


# Figure 5 in the paper
def plot_excitation_signals():
    M = 5
    fs = 10000
    fe = 20000
    T = 0.1
    Tb = T / M
    # The R matrix from step 4
    # R = [[4, 6, 2, 1, 5, 3],
    #      [3, 4, 6, 2, 1, 5],
    #      [5, 3, 4, 6, 2, 1],
    #      [1, 5, 3, 4, 6, 2],
    #      [2, 1, 5, 3, 4, 6],
    #      [6, 2, 1, 5, 3, 4]]
    #
    # The R matrix that reproduces figure 5
    # R = [[3, 4, 1, 6, 2, 5],
    #      [2, 3, 6, 5, 1, 4],
    #      [1, 2, 5, 4, 6, 3],
    #      [6, 1, 4, 3, 5, 2],
    #      [5, 6, 3, 2, 4, 1],
    #      [4, 5, 2, 1, 3, 6]]

    R = [[5, 1, 3, 4, 2],
         [3, 5, 2, 1, 4],
         [2, 3, 4, 5, 1],
         [1, 4, 5, 2, 3],
         [4, 2, 1, 3, 5]]

    for r in R:
        plt.figure()
        for i, m in enumerate(r):
            fsm = fs + ((m - 1) * (fe - fs)) / M
            fem = fsm + (fe - fs) / M

            t = np.arange(0, int(Tb * fs)) / fs
            if m % 2 == i % 2:
                y = np.linspace(fem, fsm, len(t))
            else:
                y = np.linspace(fsm, fem, len(t))
            plt.plot(t + (i * Tb), y)
            # print(m)
            # plt.show()

        plt.xlim(0, T)
        plt.grid()
        plt.ylabel("frequency [Hz]")
        plt.xlabel("time [s]")

    plt.tight_layout()


def get_orthogonal_chirps():
    M = 4
    fs = 100
    fe = 1000
    # The R matrix from step 4
    # R = [[4, 6, 2, 1, 5, 3],
    #      [3, 4, 6, 2, 1, 5],
    #      [5, 3, 4, 6, 2, 1],
    #      [1, 5, 3, 4, 6, 2],
    #      [2, 1, 5, 3, 4, 6],
    #      [6, 2, 1, 5, 3, 4]]
    #
    # The R matrix that reproduces figure 5
    # R = [[3, 4, 1, 6, 2, 5],
    #      [2, 3, 6, 5, 1, 4],
    #      [1, 2, 5, 4, 6, 3],
    #      [6, 1, 4, 3, 5, 2],
    #      [5, 6, 3, 2, 4, 1],
    #      [4, 5, 2, 1, 3, 6]]

    # R matrix for m=4
    R = [[3, 2, 1, 4],
         [2, 4, 3, 1],
         [1, 3, 4, 2],
         [4, 1, 2, 3]]

    symbols = []

    for r in R:
        chirp = []
        for i, m in enumerate(r):
            fsm = fs + ((m - 1) * (fe - fs)) / M
            fem = fsm + (fe - fs) / M

            if m % 2 == i % 2:
                chirp.append((fem, fsm))
            else:
                chirp.append((fsm, fem))

        symbols.append(chirp)

    return symbols


def convert_bit_to_chrirp(symbols, bit, M: int = 4, T: float = 0.1, no_window: bool = False) -> np.ndarray:
    # Symbols is the list of symbols we have at our disposal
    # Bit may only be 1/0
    fsample = 44100
    Tb = T / M
    t = np.linspace(0, Tb, int(np.ceil(Tb * fsample)))

    def encode(symbol):
        c = np.array([])
        for subchirp in symbol:
            # Maybe shape the chirp with a window function, to reduce the phase difference click
            if not no_window:
                window_kaiser = np.kaiser(len(t), beta=5.25)
            else:
                window_kaiser = 1
            subchirp_signal = window_kaiser * chirp(t, subchirp[0], Tb, subchirp[1])
            c = np.append(c, subchirp_signal)
        return c

    if bit == 0:
        return encode(symbols[0])
    elif bit == 1:
        return encode(symbols[1])
    else:
        print("bit is not 1 or 0!")


def get_chirps_from_bits(symbols, bits) -> [np.ndarray]:
    # Symbols is the list of symbols we have at our disposal
    # Bit must be a list of 1s and 0s
    chirps = []

    print(f"Converting {bits} to data")
    print(f"Available symbols: {symbols}")

    for bit in bits:
        chirps.append(convert_bit_to_chrirp(symbols, bit))

    return chirps


def convert_data_to_soundfile(symbols: list, data: str) -> (str, np.ndarray):
    filename = "temp.wav"
    fsample = 44100

    print(f"raw data: {data}")
    bits_to_send = tobits(data)
    chirps = get_chirps_from_bits(symbols, bits_to_send)

    # preamble = get_preamble()
    preamble = np.array([])

    from scipy.io.wavfile import write
    concat_samples = np.array(preamble)
    for sample in chirps:
        # sample = np.append(sample, np.zeros(sample.size))
        concat_samples = np.append(concat_samples, np.array(sample))
    concat_samples = concat_samples * np.iinfo(np.int16).max
    write(filename, fsample, concat_samples.astype(np.int16))
    # sd.play(concat_samples, fsample, blocking=True)

    return filename, concat_samples


def plot_overview():
    # plot_up_chirps()
    # plt.figure()
    # plot_up_chirps()
    # plt.show()
    # plt.gca().clear()
    # plot_down_chirps()
    # plt.grid(True)
    # plt.figure()
    # plot_hybrid_chirps()
    # plot_unity_pulses()
    # plot_excitation_signals()
    # plot_both()

    # Currently doing hybrid symbols, not sure about the performance difference
    # Only getting the first 2 of the 6 symbols
    symbols = get_orthogonal_chirps()[:2]

    data_to_send = "Hello, World!"

    file, data = convert_data_to_soundfile(symbols, data_to_send)
    # sd.play(data, 44100, blocking=True)

    symbol0 = convert_bit_to_chrirp(symbols, 0)
    symbol1 = convert_bit_to_chrirp(symbols, 1)
    symbol_merged = symbol0 + symbol1
    data = (data[:int(0.12 * 44100)] / np.iinfo(np.int16).max).astype(np.float64)
    conv_0 = np.convolve(data, np.flip(symbol0))
    conv_1 = np.convolve(data, np.flip(symbol1))
    conv_merged_0 = np.convolve(symbol_merged, np.flip(symbol0))
    conv_merged_1 = np.convolve(symbol_merged, np.flip(symbol1))

    fig, axs = plt.subplots(8, figsize=(8, 10))
    axs[0].plot(data)
    axs[0].set_title("[A] First symbol (data)")

    axs[1].plot(symbol0)
    axs[1].set_title("[B] Symbol 0")

    axs[2].plot(symbol1)
    axs[2].set_title("[C] Symbol 1")

    axs[3].plot(conv_0)
    # axs[3].set_ylim(10, 1000)
    axs[3].set_title("[D] Convolution result with symbol 0")
    # axs[3].set_yscale("log")

    axs[4].plot(conv_1)
    # axs[4].set_ylim(10, 1000)
    axs[4].set_title("[E] Convolution result with symbol 1")
    # axs[4].set_yscale("log")

    axs[5].plot(symbol_merged)
    axs[5].set_title("[F] Symbol 0+1 (merged)")

    axs[6].plot(conv_merged_0)
    axs[6].set_title("[G] Convolution result of Symbol 0+1 (merged) with Symbol 0")

    axs[7].plot(conv_merged_1)
    axs[7].set_title("[H] Convolution result of Symbol 0+1 (merged) with Symbol 1")

    plt.tight_layout()
    plt.show()


def get_symbols(no_window: bool = False) -> list:
    symbols = get_orthogonal_chirps()[:2]
    symbol0 = np.conjugate(np.flip(convert_bit_to_chrirp(symbols, 0, no_window=no_window)))
    symbol1 = np.conjugate(np.flip(convert_bit_to_chrirp(symbols, 1, no_window=no_window)))
    return [symbol0, symbol1]


def get_preamble(flipped: bool = False) -> np.ndarray:
    preamble = [(1000, 2000)]  # Must be outside of the regular frequencies
    preamble = convert_bit_to_chrirp([preamble], 0, M=1, T=0.025)
    if flipped:
        preamble = np.conjugate(np.flip(preamble))
    return preamble


def contains_preamble(data: np.ndarray, plot: bool = False) -> bool:
    preamble = get_preamble(True)

    if data.size == 0:
        return False

    conv_data = get_conv_results(data, [preamble])[0]

    # TODO: Is there a better way to differentiate from noise?
    peak_threshold = 10 ** 5

    if plot:
        fig, axs = plt.subplots(2)
        fig.suptitle("preamble data")
        axs[0].plot(data)
        axs[1].plot(conv_data)
        axs[1].hlines(peak_threshold, 0, data.size, color='black')

    return np.max(conv_data) > peak_threshold


def get_conv_results(data: np.ndarray, symbols: list) -> list:
    from scipy.signal import hilbert

    conv_data = []
    for symbol in symbols:
        conv_temp = np.convolve(data, symbol, mode="same")
        conv_envelope = np.abs(hilbert(conv_temp))
        conv_data.append(conv_envelope)
    return conv_data


def get_peaks(conv_data: list, plot: bool = False) -> list:
    T = 0.025
    peak_time = T * 0.05
    fsample = 44100
    peak_length = int(peak_time * fsample)
    avg_distance = int(T * fsample)
    threshold = np.max(conv_data) * 0.75  # TODO: is this a good, robust threshold?

    fig = None
    axs = None
    if plot:
        fig, axs = plt.subplots(2)
        fig.suptitle("Peaks data")

    def get_peaks_from_conv(data: np.ndarray, threshold, peak_len, ax=None) -> list:
        # First, clear the first bit of data, due to convolution artifacts
        # data[0:peak_len] = 0
        # data[-peak_len:] = 0

        # Then, find the first point above the threshold, set this as the first peak
        try:
            peak = np.where(data > threshold)[0][0]
        except IndexError:
            return []

        peaks = []
        while True:
            # Create a search range around the peak
            start = peak - peak_len
            end = peak + peak_len
            if start < 0:
                start = 0
            if end > data.size - 1:
                end = data.size - 1

            peak_range = data[start:end]

            # If this range is empty, then we're at the end
            if peak_range.size == 0:
                break

            # Get the index of the actual peak
            actual_peak = peak + np.argmax(peak_range) - peak_len

            # If this peak is above our threshold, put it in the list
            if data[actual_peak] > threshold:
                # Go to the predicted next peak based on the actual peak
                peak = actual_peak + avg_distance
                peaks.append(actual_peak)
            else:
                if plot:  # and 332000 < peak < 333000:
                    # print(f"peak: {peak} (y={data[peak]}) actual_peak: {actual_peak} (y={data[actual_peak]})")
                    ax.vlines(peak - peak_len, 0, np.max(data), color="black")
                    ax.vlines(peak + peak_len, 0, np.max(data), color="black")
                    try:
                        ax.plot(peak, data[peak], color="black", marker="D")
                    except IndexError:
                        pass
                    ax.plot(actual_peak, data[actual_peak], color="purple", marker="x")
                # Go to the predicted next peak based on the extrapolated peak
                peak = peak + avg_distance

        return peaks

    if not plot:
        peaks_s0 = get_peaks_from_conv(conv_data[0], threshold, peak_length)
        peaks_s1 = get_peaks_from_conv(conv_data[1], threshold, peak_length)
    else:
        peaks_s0 = get_peaks_from_conv(conv_data[0], threshold, peak_length, axs[0])
        peaks_s1 = get_peaks_from_conv(conv_data[1], threshold, peak_length, axs[1])

    if plot:
        axs[0].plot(conv_data[0], alpha=0.5)
        axs[1].plot(conv_data[1], alpha=0.5)
        axs[0].plot(peaks_s0, conv_data[0][peaks_s0], "xr")
        axs[1].plot(peaks_s1, conv_data[1][peaks_s1], "xb")
        axs[0].hlines(threshold, 0, conv_data[0].size, colors="black")
        axs[1].hlines(threshold, 0, conv_data[0].size, colors="black")

    return [peaks_s0, peaks_s1]


def iterative_decode(data: np.ndarray, symbols: list) -> (np.ndarray, list, list):
    plot = False
    if symbols[0].size > data.size:
        missing_data = symbols[0].size * 4 - data.size
        data = np.append(data, np.zeros(missing_data))
        plot = True

    conv = get_conv_results(data, symbols)
    peaks = get_peaks(conv, plot=plot)

    if peaks == [[], []]:
        return np.array([]), conv, peaks

    last_peak = np.max(np.amax(peaks))
    # Conv peak is in the middle, so add half a symbol
    # this seems to break stuff
    # But, we need it for phase shift decoding.
    # data_after_peak = data[last_peak:]
    data_after_peak = np.array([])

    if plot:
        plt.figure()
        plt.plot(data)

    return data_after_peak, conv, peaks


def get_bits_from_peaks(peaks: list) -> list:
    # Mark the respective symbols and merge them in a sorted list
    peaks_s0 = list(map(lambda x: (x, 0), peaks[0]))
    peaks_s1 = list(map(lambda x: (x, 1), peaks[1]))
    symbol_peaks = peaks_s0 + peaks_s1
    symbol_peaks.sort()

    # Reconstruct the data
    bits = []
    for symbol in symbol_peaks:
        bits.append(symbol[1])

    return bits


def decode_file(data: np.ndarray, plot: bool = False, iterative: bool = True):
    # Get the original symbols, used for convolution
    symbols = get_symbols(no_window=True)

    if iterative:
        T = 0.025
        fsample = 44100
        n = 13
        CHUNK = int(n * T * fsample)

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        peaks = [[], []]
        conv_data = [[], []]
        data_after_peak = np.array([])
        processed_data = np.array([])
        preamble_found = False
        for i, chunk_data in enumerate(chunks(data, CHUNK)):

            # First, wait for preamble
            if not preamble_found and contains_preamble(chunk_data, plot=False):
                print("preamble found!")
                preamble_found = True

            # We sometimes have a little bit of data left at the end (rounding?)
            # Just ignore it
            if len(chunk_data) < CHUNK // n:
                print("ignoring last bit of data")
                continue

            # Append the data after the last peak to the new data
            data_to_search = np.append(data_after_peak, chunk_data)

            data_after_peak, data_to_search_conv, data_to_search_peaks = iterative_decode(data_to_search, symbols)

            # print(data_to_search_peaks)
            # print(len(processed_data))

            # Offset the peak on all the received data
            peak_offset = len(processed_data)
            data_to_search_peaks[0] = list(map(lambda x: x + peak_offset, data_to_search_peaks[0]))
            data_to_search_peaks[1] = list(map(lambda x: x + peak_offset, data_to_search_peaks[1]))

            # Accumulate the data for plotting
            peaks[0] = peaks[0] + data_to_search_peaks[0]
            peaks[1] = peaks[1] + data_to_search_peaks[1]
            conv_data[0] = np.append(conv_data[0], data_to_search_conv[0])
            conv_data[1] = np.append(conv_data[1], data_to_search_conv[1])

            # print(peaks)

            # # Remove duplicates, should not happen
            # from collections import Counter
            # print(f"doubles[0]: {[item for item, count in Counter(peaks[0]).items() if count > 1]}")
            # print(f"doubles[1]: {[item for item, count in Counter(peaks[1]).items() if count > 1]}")
            # peaks[0] = list(np.unique(peaks[0]))
            # peaks[1] = list(np.unique(peaks[1]))

            # Add the data that has been processed (before the last peak) to the processed data array
            # For some reason [:-0] does not give the entire array, but gives []
            if data_after_peak.size != 0:
                processed_data = np.append(processed_data, data_to_search[:-data_after_peak.size])
            else:
                processed_data = np.append(processed_data, data_to_search)

            bits = get_bits_from_peaks(peaks)

            received_data = frombits(bits)
            print(received_data)

        peaks[0] = peaks[0][:-1]

        # Append final data
        processed_data = np.append(processed_data, data_after_peak)

        # Get conv data for plotting
        # conv_data = get_conv_results(processed_data, symbols)
    else:
        if contains_preamble(data, plot=plot):
            print("preamble found!")

        # For compatibility with iterative decode
        processed_data = data

        # Convolve the data with the original symbols, this produces periodic peaks
        conv_data = get_conv_results(data, symbols)

        # Find the peaks
        peaks = get_peaks(conv_data, plot=plot)

        bits = get_bits_from_peaks(peaks)

        received_data = frombits(bits)
        print("resulting data:")
        print(received_data)
        print(bits)

    # Plot the results
    if plot:
        fig, axs = plt.subplots(4)
        fig.suptitle("decode file results")
        axs[0].plot(data)
        axs[0].plot(processed_data, alpha=0.5)
        axs[1].plot(conv_data[0])
        axs[2].plot(conv_data[1])
        axs[1].plot(peaks[0], conv_data[0][peaks[0]], "xr")
        axs[2].plot(peaks[1], conv_data[1][peaks[1]], "xb")
        axs[3].plot(get_conv_results(processed_data, symbols)[0])
        axs[3].plot(get_conv_results(processed_data, symbols)[1])

    return received_data, bits


def calculate_ber(send_data: list, received_data: list) -> float:
    err = 0
    for i, bit in enumerate(send_data):
        try:
            print(f"{bit}:{received_data[i]}")
            if bit != received_data[i]:
                err += 1
        except IndexError:
            print(f"{bit}:???")

    if len(send_data) != len(received_data):
        print(f"received bits ({len(received_data)}) not the same length as transmitted ({len(send_data)})!")

    return err / len(send_data)


def append_zeros(data: np.ndarray) -> np.ndarray:
    fsample = 44100
    T = 0.025
    symbol_size = int(T * fsample)
    random = np.random.randint(0, 10 * symbol_size)
    return np.append(np.zeros(random), data)


from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal


def add_wgn(s, SNRdB, L=1):
    """
    # author - Mathuranathan Viswanathan (gaussianwaves.com
    # This code is part of the book Digital Modulations using Python
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    N0 = P / gamma  # Find the noise spectral density
    if isrealobj(s):  # check if input is real/complex object type
        n = sqrt(N0 / 2) * standard_normal(s.shape)  # computed noise
    else:
        n = sqrt(N0 / 2) * (standard_normal(s.shape) + 1j * standard_normal(s.shape))
    r = s + n  # received signal
    return r


def run_offline_test(file: str = None):
    import sounddevice as sd
    # Currently doing hybrid symbols, I think the paper concluded that there where no performance differences
    # Only getting the first 2 of the 6 symbols, since we only need to transmit 2 symbols
    symbols = get_orthogonal_chirps()[:2]

    data_to_send = "Hello, World!"

    if file is None:
        file, data = convert_data_to_soundfile(symbols, data_to_send)
        fs = 44100
    else:
        from scipy.io.wavfile import read
        fs, data = read(file)
    # sd.play(data, fs, blocking=True)

    # .. we can now do with the data what we want
    # For example, add some zeros to it
    # data = append_zeros(data)
    # And add noise to it
    # data = add_wgn(data, snr=-10)

    # Now, we attempt to decode it
    received_data, received_bits = decode_file(data, plot=True)
    ber = calculate_ber(tobits(data_to_send), received_bits)

    print(f"BER: {ber}")


def run_live_test():
    import pyaudio
    T = 0.025
    fsample = 44100
    n = 2
    CHUNK = int(n * T * fsample)

    data_to_send = "Hello, World!"
    bits_to_send = tobits(data_to_send)
    data_len = len(bits_to_send)

    print(f"live test, searching for : {data_to_send}")

    symbols = get_symbols()

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fsample, input=True, frames_per_buffer=CHUNK)

    all_data = np.array([])
    tempdata = np.array([])
    import time
    start = time.time()
    while time.time() - start < 3:
        tempdata = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        if contains_preamble(tempdata):
            print("preamble found!")
            all_data = tempdata
            break

    # We can maybe decode this live if we can synchronize the peaks and know when we've gotten a full symbol
    # However, why would we want this?
    for _ in range(data_len // n + 1):
        tempdata = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        all_data = np.append(all_data, tempdata)

    print("Finished recording")

    conv_data = get_conv_results(all_data, symbols)

    fig, axs = plt.subplots(3)
    fig.suptitle("Live decode")
    axs[0].plot(all_data)
    axs[1].plot(conv_data[0])
    axs[1].plot(conv_data[1])

    received_data, received_bits = decode_file(all_data, plot=True)
    ber = calculate_ber(tobits(data_to_send), received_bits)
    print(f"ber: {ber}")

    from scipy.io.wavfile import write
    write("microphone.wav", fsample, all_data.astype(np.int16))


def main():
    plot_overview()
    # run_offline_test()
    # run_live_test()

    plt.show()


if __name__ == "__main__":
    main()
