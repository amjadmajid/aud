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
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


# Figure 3.a in the paper
def plot_up_chirps():
    M = 6
    fs = 38000
    fe = 44000
    T = 0.012
    Tb = T/M
    for m in range(M):
        fsm = fs + (m*(fe - fs))/M
        fem = fsm + (fe - fs)/M

        t = np.arange(0, int(Tb * fs)) / fs
        y = np.linspace(fsm, fem, len(t))

        for m_ in range(M):
            plt.plot(t + (m_ * Tb), y)

    plt.xlim(0, T)
    plt.grid()


# Figure 3.b in the paper
def plot_down_chirps():
    M = 6
    fs = 38000
    fe = 44000
    T = 0.012
    Tb = T/M
    for m in range(M):
        fsm = fs + (m*(fe - fs))/M
        fem = fsm + (fe - fs)/M

        t = np.arange(0, int(Tb * fs)) / fs
        y = np.linspace(fem, fsm, len(t))

        for m_ in range(M):
            plt.plot(t + (m_ * Tb), y)

    plt.xlim(0, T)
    plt.grid()


# Figure 3.c in the paper
def plot_hybrid_chirps():
    M = 6
    fs = 38000
    fe = 44000
    T = 0.012
    Tb = T/M
    for m in range(1, M+1):
        fsm = fs + ((m-1)*(fe - fs))/M
        fem = fsm + (fe - fs)/M

        t = np.arange(0, int(Tb * fs)) / fs
        y = np.linspace(fsm, fem, len(t))
        y_not = np.linspace(fem, fsm, len(t))

        start = int(m % 2 == 0)
        for i in range(start, M+start):
            if i % 2 == 0:
                plt.plot(t + ((i-start) * Tb), y)
            else:
                plt.plot(t + ((i-start) * Tb), y_not)

    plt.xlim(0, T)
    plt.grid()


# Figure 4a-f in the paper
def plot_unity_pulses():
    M = 6
    T = 0.012
    Tb = T/M
    R = [[4, 6, 2, 1, 5, 3],
         [3, 4, 6, 2, 1, 5],
         [5, 3, 4, 6, 2, 1],
         [1, 5, 3, 4, 6, 2],
         [2, 1, 5, 3, 4, 6],
         [6, 2, 1, 5, 3, 4]]

    for r in R:
        fig, axs = plt.subplots(M, sharex=True, sharey=True)
        for i, element in enumerate(r):
            start = (element-1)*Tb
            end = start + Tb
            x = np.linspace(0, T, 1000)
            y = np.logical_and(x >= start, x <= end)
            axs[i].plot(x, y)
            axs[i].set_ylim(-0.1, 1.1)
        fig.tight_layout()

    plt.show()


# Figure 5 in the paper
def plot_excitation_signals():
    M = 6
    T = 0.012
    fs = 3800
    fe = 44000
    Tb = T/M
    # The R matrix from step 4
    # R = [[4, 6, 2, 1, 5, 3],
    #      [3, 4, 6, 2, 1, 5],
    #      [5, 3, 4, 6, 2, 1],
    #      [1, 5, 3, 4, 6, 2],
    #      [2, 1, 5, 3, 4, 6],
    #      [6, 2, 1, 5, 3, 4]]
    #
    # The R matrix that reproduces figure 5
    R = [[3, 4, 1, 6, 2, 5],
         [2, 3, 6, 5, 1, 4],
         [1, 2, 5, 4, 6, 3],
         [6, 1, 4, 3, 5, 2],
         [5, 6, 3, 2, 4, 1],
         [4, 5, 2, 1, 3, 6]]

    for r in R:
        plt.figure()
        for i, m in enumerate(r):
            fsm = fs + ((m-1)*(fe - fs))/M
            fem = fsm + (fe - fs)/M

            t = np.arange(0, int(Tb * fs)) / fs
            if m % 2 == i % 2:
                y = np.linspace(fem, fsm, len(t))
            else:
                y = np.linspace(fsm, fem, len(t))
            plt.plot(t + (i * Tb), y)

        plt.xlim(0, T)
        plt.grid()

    plt.tight_layout()


def get_orthogonal_chirps():
    M = 4
    fs = 2000
    fe = 6000
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
            fsm = fs + ((m-1)*(fe - fs))/M
            fem = fsm + (fe - fs)/M

            if m % 2 == i % 2:
                chirp.append((fem, fsm))
            else:
                chirp.append((fsm, fem))

        symbols.append(chirp)

    return symbols


def convert_bit_to_chrirp(symbols, bit) -> np.ndarray:
    # Symbols is the list of symbols we have at our disposal
    # Bit may only be 1/0
    M = 4
    T = 0.12
    fsample = 44100
    Tb = T / M
    t = np.linspace(0, Tb, int(np.ceil(Tb * fsample)))

    def encode(symbol):
        c = np.array([])
        for subchirp in symbol:
            # Maybe shape the chirp with a window function, to reduce the phase difference click
            window_kaiser = np.kaiser(len(t), beta=5.25)
            # window_kaiser = 1
            subchirp_signal = window_kaiser*chirp(t, subchirp[0], Tb, subchirp[1])
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

    s0 = convert_bit_to_chrirp(symbols, 0)
    s1 = convert_bit_to_chrirp(symbols, 1)
    # s_preamble = s0 + s1
    # s_preamble = np.append(s_preamble, np.zeros(s_preamble.size))
    s_preamble = np.array([])

    from scipy.io.wavfile import write
    concat_samples = np.array(s_preamble)
    for sample in chirps:
        # sample = np.append(sample, np.zeros(sample.size))
        concat_samples = np.append(concat_samples, np.array(sample))
    concat_samples = concat_samples * np.iinfo(np.int16).max
    write(filename, fsample, concat_samples.astype(np.int16))
    # sd.play(concat_samples, fsample, blocking=True)

    return filename, concat_samples


def plot_overview():
    # plot_up_chirps()
    # plot_down_chirps()
    # plot_hybrid_chirps()
    # plot_unity_pulses()
    # plot_excitation_signals()

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
    conv_merged_0 = np.convolve(symbol_merged, np.conjugate(np.flip(symbol0)))
    conv_merged_1 = np.convolve(symbol_merged, np.conjugate(np.flip(symbol1)))

    fig, axs = plt.subplots(8, figsize=(8, 10))
    axs[0].plot(data)
    axs[0].set_title("[A] First symbol (data)")

    axs[1].plot(symbol0)
    axs[1].set_title("[B] Symbol 0")

    axs[2].plot(symbol1)
    axs[2].set_title("[C] Symbol 1")

    axs[3].plot(conv_0)
    axs[3].set_ylim(-700, 700)
    axs[3].set_title("[D] Convolution result with symbol 0")

    axs[4].plot(conv_1)
    axs[4].set_ylim(-700, 700)
    axs[4].set_title("[E] Convolution result with symbol 1")

    axs[5].plot(symbol_merged)
    axs[5].set_title("[F] Symbol 0+1 (merged)")

    axs[6].plot(conv_merged_0)
    axs[6].set_title("[G] Convolution result of Symbol 0+1 (merged) with Symbol 0")

    axs[7].plot(conv_merged_1)
    axs[7].set_title("[H] Convolution result of Symbol 0+1 (merged) with Symbol 1")

    plt.tight_layout()
    plt.show()


def is_preamble(peaks: list) -> bool:
    fsample = 44100
    T = 0.12
    max_distance = T * 0.1  # 12ms
    max_diff_samples = int(fsample * max_distance)

    if len(peaks) < 2:
        print(f"only {len(peaks)} peaks in preamble")
        return False

    p0 = peaks[0]
    p1 = peaks[1]

    print(p0)
    print(p1)

    # If they are the same symbol, then it cannot be the preamble
    if p0[1] == p1[1]:
        # print("same symbol")
        return False
    # If the peaks are not 'close', then they are not part of the preamble
    elif np.abs(p1[0] - p0[0]) > max_diff_samples:
        # print("Too far apart")
        return False
    # If the previous statements did not return, then this is the preamble
    else:
        return True


def remove_preamble(peaks: list) -> list:
    num_preamble = 1
    for i in range(2, 2 + num_preamble*2, 2):
        if is_preamble(peaks[:2]):
            peaks = peaks[i:]

    return peaks


def get_symbols() -> list:
    symbols = get_orthogonal_chirps()[:2]
    symbol0 = np.conjugate(np.flip(convert_bit_to_chrirp(symbols, 0)))
    symbol1 = np.conjugate(np.flip(convert_bit_to_chrirp(symbols, 1)))
    return [symbol0, symbol1]


def get_conv_results(data: np.ndarray, symbols: list) -> list:
    conv_data = []
    for symbol in symbols:
        conv_data.append(np.convolve(data, symbol))
    return conv_data


def get_peaks(conv_data: list) -> list:
    T = 0.12
    fsample = 44100

    # TODO: tweak the height threshold
    # TODO: I don't know what is happening under the hood
    from scipy.signal import find_peaks
    height = np.max(conv_data) * 0.8
    distance = int(T*fsample) * 0.95
    print(height)

    # Make sure we dont detect noise
    # min_height = 10**6
    # if height < min_height:
    #     return [[], []]

    # These peaks are the x coords, can be converted to time by dividing by fsample
    peaks_s0, _ = find_peaks(conv_data[0], height=height, distance=distance)
    peaks_s1, _ = find_peaks(conv_data[1], height=height, distance=distance)

    return [peaks_s0, peaks_s1]


def decode_file(data: np.ndarray, plot: bool = False):

    # Get the original symbols, used for convolution
    symbols = get_symbols()

    # Convolve the data with the original symbols, this produces periodic peaks
    conv_data = get_conv_results(data, symbols)

    # Find the peaks
    peaks = get_peaks(conv_data)

    # Mark the respective symbols and merge them in a sorted list
    peaks_s0 = list(map(lambda x: (x, 0), peaks[0]))
    peaks_s1 = list(map(lambda x: (x, 1), peaks[1]))
    symbol_peaks = peaks_s0 + peaks_s1
    symbol_peaks.sort()
    print(symbol_peaks)

    # if is_preamble(symbol_peaks):
    #     print("preamble found!")
    #     symbol_peaks = remove_preamble(symbol_peaks)
    # else:
    #     print("preamble not found!")

    # Reconstruct the data
    bits = []
    peak_threshold = 2 * 10**4
    for symbol in symbol_peaks:
        value = conv_data[symbol[1]][symbol[0]]
        if value > peak_threshold:  # Make sure we don't interpret noise as data
            bits.append(symbol[1])

    received_data = frombits(bits)
    print("resulting data:")
    print(received_data)
    print(bits)

    # Plot the results
    if plot:
        fig, axs = plt.subplots(3)
        axs[0].plot(data)
        axs[1].plot(conv_data[0])
        axs[2].plot(conv_data[1])
        axs[1].plot(peaks[0], conv_data[0][peaks[0]], "xr")
        axs[2].plot(peaks[1], conv_data[1][peaks[1]], "xb")
        plt.show()

    return received_data, bits


def calculate_ber(send_data: list, received_data: list) -> float:
    err = 0
    for i, bit in enumerate(send_data):
        try:
            print(f"{bit}:{received_data[i]}")
            if bit != received_data[i]:
                err += 1
        except IndexError:
            break
    return err/len(send_data)


def append_zeros(data: np.ndarray) -> np.ndarray:
    fsample = 44100
    T = 0.12
    symbol_size = int(T * fsample)
    random = np.random.randint(0, 10*symbol_size)
    return np.append(np.zeros(random), data)


def add_wgn(data: np.ndarray, snr: float) -> np.ndarray:
    sigpower = sum([pow(abs(data[i]), 2) for i in range(data.size)])
    sigpower = sigpower / data.size
    noisepower = sigpower / pow(10, snr / 10)
    noise = np.sqrt(noisepower) * np.random.uniform(-1, 1, size=data.size)
    return data + noise


def wait_for_preamble(data: np.ndarray) -> (bool, list):
    # Get the original symbols, used for convolution
    symbols = get_symbols()

    # Convolve the data with the original symbols, this produces periodic peaks
    conv_data = get_conv_results(data, symbols)

    # Find the peaks
    peaks = get_peaks(conv_data)

    if len(peaks[0]) == 0 or len(peaks[1]) == 0:
        # print("no peaks")
        return False, []

    # plt.plot(data)
    # plt.plot(conv_data_s0)
    # plt.plot(conv_data_s1)
    # plt.plot(peaks_s0, conv_data[0][peaks_s0], "xr")
    # plt.plot(peaks_s1, conv_data[1][peaks_s1], "xb")
    # plt.show()

    # Mark the respective symbols and merge them in a sorted list
    peaks_s0 = list(map(lambda x: (x, 0), peaks[0]))
    peaks_s1 = list(map(lambda x: (x, 1), peaks[1]))
    symbol_peaks = peaks_s0 + peaks_s1
    symbol_peaks.sort()

    if is_preamble(symbol_peaks):
        print("preamble found!")
        return True, symbol_peaks
    else:
        # print("preamble not found!")
        return False, []


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
    T = 0.12
    fsample = 44100
    CHUNK = int(2 * T * fsample)

    data_to_send = "Hello, World!"
    bits_to_send = tobits(data_to_send)
    data_len = len(bits_to_send)

    print(f"live test, searching for : {data_to_send}")

    symbols = get_symbols()

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fsample, input=True,frames_per_buffer=CHUNK)

    data = np.array([])
    import time
    start = time.time()
    while time.time() - start < 15:
        tempdata = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        data = np.append(data, tempdata)

        # received_data, received_bits = decode_file(data)
        # print(received_data)

    print("done recording")

    conv_data = get_conv_results(data, symbols)

    fig, axs = plt.subplots(3)
    axs[0].plot(data)
    axs[1].plot(conv_data[0])
    axs[1].plot(conv_data[1])
    plt.show()

    from scipy.io.wavfile import write
    write("microphone.wav", fsample, data.astype(np.int16))


def main():
    # plot_overview()
    run_offline_test("microphone_no_preamble_no_break_with_window.wav")
    # run_live_test()


if __name__ == "__main__":
    main()
