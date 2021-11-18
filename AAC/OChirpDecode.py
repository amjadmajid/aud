import numpy as np
from BitManipulation import frombits, tobits
from OChirpEncode import OChirpEncode
from scipy.signal import hilbert
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
# from butterworth import butter_bandpass_filter


class OChirpDecode:

    def __init__(self, original_data: str, encoder: OChirpEncode):
        self.__encoder = encoder

        self.T = encoder.T
        self.fsample = encoder.fsample

        self.original_data = original_data
        self.original_data_bits = tobits(original_data)

        # TODO: Is there a better way to differentiate from noise?
        self.__preamble_min_peak = 10**5

    def get_preamble(self, flipped: bool = True) -> np.ndarray:
        return self.__encoder.get_preamble(flipped)

    first = True

    def get_symbols(self, no_window: bool = False) -> list:
        symbols = self.__encoder.get_orthogonal_chirps()
        symbol0 = np.conjugate(np.flip(self.__encoder.convert_bit_to_chrirp(symbols, 0, no_window=no_window,
                                                                            blank_space=False,
                                                                            T=self.T-self.__encoder.blank_space_time,
                                                                            minimal_sub_chirp_duration=self.__encoder.minimal_sub_chirp_duration)))
        symbol1 = np.conjugate(np.flip(self.__encoder.convert_bit_to_chrirp(symbols, 1, no_window=no_window,
                                                                            blank_space=False,
                                                                            T=self.T-self.__encoder.blank_space_time,
                                                                            minimal_sub_chirp_duration=self.__encoder.minimal_sub_chirp_duration)))

        if self.first is False:
            fig, axs = plt.subplots(2)
            fig.suptitle(f"{self.__encoder.fs/1000:.0f}-{self.__encoder.fe/1000:.0f}kHz, T={self.T/1000:.1f} ms")
            axs[0].plot(symbol0)
            axs[0].set_title("Symbol 0")
            axs[1].plot(symbol1)
            axs[1].set_title("Symbol 1")
            fig.tight_layout()
            self.first = True

        return [symbol0, symbol1]

    @staticmethod
    def get_conv_results(data: np.ndarray, symbols: list) -> list:
        from scipy.signal import oaconvolve
        conv_data = []
        for symbol in symbols:
            # conv_temp = np.convolve(data, symbol, mode="same")
            # This convolution is much faster than np.convolve
            # Best when one input is very large (n>500) and the other is small
            conv_temp = oaconvolve(data, symbol, mode="same")
            conv_envelope = np.abs(hilbert(conv_temp))
            conv_data.append(conv_envelope)

        return conv_data

    def get_peaks(self, data: list, plot: bool = False) -> list:

        # Typical distance between peaks, to predict the next peak
        avg_distance = int(self.T * self.fsample)

        # Define a peak time to search for the actual peak around the predicted peak
        peak_time = self.T * 0.05
        peak_length = int(peak_time * self.fsample)

        if data[0].size != data[1].size:
            print("get peaks data sizes not the same!")
            return []

        # TODO: is this a good, robust threshold?
        # Only used to determine the first peak, should be above the noise floor
        # And also above the cross correlation peak
        # threshold = (np.mean(data) + np.max(data)) / 2
        threshold = np.max(data) * 0.75

        def get_first_peak(data: np.ndarray, threshold: float) -> int:
            try:
                return np.where(data > threshold)[0][0]
            except IndexError:
                return np.iinfo(int).max

        # First, find the first point above the threshold, assume this to be the first peak
        peaks0 = get_first_peak(data[0], threshold)
        peaks1 = get_first_peak(data[1], threshold)
        peak = min(peaks0, peaks1)

        if peak == np.iinfo(int).max:
            print("Did not find peak!")
            return []

        if plot:
            fig, axs = plt.subplots(2, sharex=True)
            fig.suptitle("Peaks data")
        else:
            axs = None

        peaks = []
        while True:
            # Create a search range around the peak
            start = peak - peak_length
            end = peak + peak_length
            if start < 0:
                start = 0
            if end > data[0].size - 1:
                end = data[0].size - 1

            peak_range_s0 = data[0][start:end]
            peak_range_s1 = data[1][start:end]

            # If this range is empty, then we're at the end
            if peak_range_s0.size == 0:
                break

            # Get the index of the actual peaks
            actual_peak_s0 = peak + np.argmax(peak_range_s0) - peak_length
            actual_peak_s1 = peak + np.argmax(peak_range_s1) - peak_length

            # We take the highest peak of the two symbols as the actual symbol
            # We also assume it is not noise, since we have the preamble
            highest_peak, symbol = (actual_peak_s0, 0) if data[0][actual_peak_s0] > data[1][actual_peak_s1] else (actual_peak_s1, 1)
            peaks.append((highest_peak, symbol))

            # Plot some debug information
            if plot:
                axs[0].vlines(peak - peak_length, 0, np.max(data), color="black", alpha=0.5)
                axs[0].vlines(peak + peak_length, 0, np.max(data), color="black", alpha=0.5)
                axs[1].vlines(peak - peak_length, 0, np.max(data), color="black", alpha=0.5)
                axs[1].vlines(peak + peak_length, 0, np.max(data), color="black", alpha=0.5)
                axs[symbol].plot(highest_peak, data[symbol][highest_peak], color="purple", marker="x")
                try:
                    axs[0].plot(peak, data[0][peak], color="black", marker="D")
                    axs[1].plot(peak, data[1][peak], color="black", marker="D")
                except IndexError:
                    pass

            # Go to the predicted next peak based on the actual peak
            peak = highest_peak + avg_distance

        if plot:
            axs[0].plot(data[0])
            axs[1].plot(data[1])
            axs[0].hlines(threshold, 0, data[0].size, colors="black")
            axs[1].hlines(threshold, 0, data[0].size, colors="black")

        return peaks

    def contains_preamble(self, data: np.ndarray, plot: bool = False) -> bool:
        preamble = self.get_preamble(True)

        if data.size == 0:
            return False

        conv_data = self.get_conv_results(data, [preamble])[0]

        if plot:
            fig, axs = plt.subplots(2)
            fig.suptitle("preamble data")
            axs[0].plot(data)
            axs[1].plot(conv_data)
            axs[1].hlines(self.__preamble_min_peak, 0, data.size, color='black')

        self.__preamble_min_peak = 10 * np.mean(conv_data)

        return np.max(conv_data) > self.__preamble_min_peak

    def iterative_decode(self, data: np.ndarray, symbols: list) -> (np.ndarray, list, list):
        plot = False
        if symbols[0].size > data.size:
            missing_data = symbols[0].size * 4 - data.size
            data = np.append(data, np.zeros(missing_data))
            plot = True

        conv = self.get_conv_results(data, symbols)
        peaks = self.get_peaks(conv, plot=plot)

        if peaks == [[], []]:
            return np.array([]), conv, peaks

        # Conv peak is in the middle, so add half a symbol
        # this seems to break stuff
        # But, we need it for phase shift decoding.
        # last_peak = np.max(np.amax(peaks))
        # data_after_peak = data[last_peak:]
        data_after_peak = np.array([])

        if plot:
            plt.figure()
            plt.plot(data)

        return data_after_peak, conv, peaks

    @staticmethod
    def get_bits_from_peaks(peaks: list) -> list:
        return list(map(lambda x: x[1], peaks))

    def calculate_ber(self, received_data: list, do_print: bool = False) -> float:

        def my_print(str, end='\n'):
            if do_print:
                print(str, end=end)

        err = 0
        for i, bit in enumerate(self.original_data_bits):
            try:
                my_print(f"{bit}:{received_data[i]}")
                if bit != received_data[i]:
                    err += 1
            except IndexError:
                err += 1
                my_print(f"{bit}:???")

        if len(self.original_data_bits) != len(received_data):
            my_print(f"received bits ({len(received_data)}) not the same length as transmitted ({len(self.original_data_bits)})!")

        ber = err / len(self.original_data_bits)

        if do_print is False and ber != 0.0:
            return self.calculate_ber(received_data, do_print=True)
        else:
            return ber

    """
    Iterative decoding allows us to decode data while still receiving. However, the decoding quality is lacking and can only
    reach the regular decoding. So the only advantage is that we can decode while receiving.
    However, this is only a test function. 
    """
    def decode_data_iterative(self, data: np.ndarray, plot: bool = False) -> (str, list):
        # Get the original symbols, used for convolution
        symbols = self.get_symbols(no_window=True)

        # How many symbols to listen for at once
        n = 13
        CHUNK = int(n * self.T * self.fsample)

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        peaks = [[], []]
        conv_data = [[], []]
        data_after_peak = np.array([])
        processed_data = np.array([])
        preamble_found = False
        received_data = "None"
        bits = []
        for i, chunk_data in enumerate(chunks(data, CHUNK)):

            # First, wait for preamble
            if not preamble_found and self.contains_preamble(chunk_data, plot=False):
                print("preamble found!")
                preamble_found = True

            # We sometimes have a little bit of data left at the end (rounding?)
            # Just ignore it
            if len(chunk_data) < CHUNK // n:
                print("ignoring last bit of data")
                continue

            # Append the data after the last peak to the new data
            data_to_search = np.append(data_after_peak, chunk_data)

            data_after_peak, data_to_search_conv, data_to_search_peaks = self.iterative_decode(data_to_search, symbols)
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

            bits = self.get_bits_from_peaks(peaks)

            received_data = frombits(bits)

        # Plot the results
        if plot:
            fig, axs = plt.subplots(4)
            fig.suptitle("decode file results")

            # Sometimes the last one is out of the index. Not sure why
            peaks[0] = peaks[0][:-1]

            # Append final data
            processed_data = np.append(processed_data, data_after_peak)

            # Get conv data for plotting
            # conv_data = get_conv_results(processed_data, symbols)

            axs[0].plot(data)
            axs[0].plot(processed_data, alpha=0.5)
            axs[1].plot(conv_data[0])
            axs[2].plot(conv_data[1])
            axs[1].plot(peaks[0], conv_data[0][peaks[0]], "xr")
            axs[2].plot(peaks[1], conv_data[1][peaks[1]], "xb")
            axs[3].plot(self.get_conv_results(processed_data, symbols)[0])
            axs[3].plot(self.get_conv_results(processed_data, symbols)[1])

        return received_data, bits

    def decode_data_regular(self, data: np.ndarray, plot: bool = False) -> (str, list):

        # Get the original symbols, used for convolution
        symbols = self.get_symbols(no_window=False)

        if self.contains_preamble(data, plot=plot):
            print("preamble found!")
        else:
            print("NO PREAMBLE FOUND!")
            return "", []

        # Convolve the data with the original symbols, this produces periodic peaks
        conv_data = self.get_conv_results(data, symbols)

        # Find the peaks
        peaks = self.get_peaks(conv_data, plot=plot)

        bits = self.get_bits_from_peaks(peaks)

        received_data = frombits(bits)

        # Plot the results
        if plot:
            fig, axs = plt.subplots(3, sharex=True)
            fig.suptitle("Decode data results")
            axs[0].set_title("Microphone data")
            axs[0].plot(data)
            axs[1].set_title("Convolution result with symbol 0")
            axs[1].plot(conv_data[0])
            axs[2].set_title("Convolution result with symbol 1")
            axs[2].plot(conv_data[1])
            for peak in peaks:
                i = peak[1] + 1
                axs[i].plot(peak[0], conv_data[peak[1]][peak[0]], "xr")
            # axs[1].plot(peaks[0], conv_data[0][peaks[0]], "xr")
            # axs[2].plot(peaks[1], conv_data[1][peaks[1]], "xb")

        return received_data, bits

    def decode_data(self, data: np.ndarray, plot: bool = False, iterative: bool = False) -> float:
        print(f"Starting decode. Should receive [{self.original_data}]")

        if iterative:
            # Iterative does not really offer any advantage, but was a nice experiment for fully-live decoding
            received_data, bits = self.decode_data_iterative(data, plot)
        else:
            received_data, bits = self.decode_data_regular(data, plot)

        ber = self.calculate_ber(bits)
        print(f"Got [{received_data}] with ber: {ber}")

        if plot:
            plt.tight_layout()
            plt.show()

        return ber

    def decode_file(self, file: str, plot: bool = False) -> float:
        from scipy.io.wavfile import read

        fs, data = read(file)
        self.fsample = fs
        return self.decode_data(data, plot=plot)

    def decode_live(self, plot: bool = True, do_not_process: bool = False) -> float:
        import pyaudio
        import time

        print(f"live test, searching for : {self.original_data}")

        n = int(self.__encoder.T_preamble / self.T)
        CHUNK = int(n * self.T * self.fsample)
        data_len = len(self.original_data_bits)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.fsample, input=True, frames_per_buffer=CHUNK)

        print("Waiting for preamble...")
        start_time = time.time()
        all_data = np.array([])
        while True:
            tempdata = np.frombuffer(stream.read(4*CHUNK), dtype=np.int16)

            # Save all data for if we time-out
            all_data = np.append(all_data, tempdata)

            if self.contains_preamble(tempdata):
                print("preamble found!")
                # Save this bit as well, since it can contain part of a symbol
                # We remove the rest because we dont need it
                all_data = tempdata
                break

            if time.time() - start_time > 15:
                print("Timeout waiting for preamble")
                break

        print("Recording rest of the message")
        tempdata = np.frombuffer(stream.read(CHUNK * (data_len // n + 1)), dtype=np.int16)
        all_data = np.append(all_data, tempdata)
        stream.close()
        print("Finished recording")

        # butter_bandpass_filter(data=all_data, lowcut=self.__encoder.fs, highcut=self.__encoder.fe, fs=self.fsample, order=1)

        # IF we just want to record the file, skip the decode function
        if not do_not_process:
            ber = self.decode_data(all_data, plot=plot)
        else:
            ber = None

        print("Finished decoding, writing result to file.")
        write("microphone.wav", self.fsample, all_data.astype(np.int16))

        return ber


if __name__ == '__main__':
    data_to_send = "Hello, World!"

    encoder = OChirpEncode(minimal_sub_chirp_duration=False)
    file, data = encoder.convert_data_to_sound(data_to_send)
    oc = OChirpDecode(original_data=data_to_send, encoder=encoder)
    oc.decode_file("temp.wav", plot=True)
    # oc.decode_live()

