import numpy as np
from BitManipulation import frombits, tobits
from scipy.signal import chirp
from scipy.io.wavfile import write


class OChirpEncode:

    def __init__(self, fsample: int = 44100, T: float = None, M: int = 4, fs: int = 100, fe: int = 1000,
                 f_preamble_start: int = 1000, f_preamble_end: int = 2000, blank_space_time: float = 0.006,
                 T_preamble: float = 0.2, orthogonal_pair_offset: int = 0, required_number_of_cycles: int = 5,
                 minimal_sub_chirp_duration: bool = False, volume: float = 1, no_window: bool = False):

        self.fsample = fsample
        self.M = M  # Hardcoded function
        self.fs = fs
        self.fe = fe
        self.preamble_start = f_preamble_start
        self.preamble_end = f_preamble_end
        self.blank_space_time = blank_space_time
        self.orthogonal_pair_offset = orthogonal_pair_offset
        self.required_number_of_cycles = required_number_of_cycles

        """
            minimal sub chirp duration optimizes the subchirps such that the length is minimal (guarantees exact 
            required_number_of_cycles. This way, we reduce the symbol time. However, we compromise on the orthogonality.
        """
        self.minimal_sub_chirp_duration = minimal_sub_chirp_duration

        self.volume = volume
        self.no_window = no_window

        if T is None:
            self.T = self.get_min_symbol_time(M, required_number_of_cycles, fs, fe, minimal_sub_chirp_duration) \
                        + blank_space_time
            self.T_preamble = T_preamble
            print(f"Calculated minimum symbol time: {self.T*1000:.1f} ms")
        elif minimal_sub_chirp_duration is False:
            min_symbol_time = self.get_min_symbol_time(M, 1, fs, fe) + blank_space_time
            if T < min_symbol_time:
                print(f"WARNING: The given T [{T*1000:.1f} ms] is smaller than required for a single cycle [{min_symbol_time*1000:.1f} ms]!")
                print("This will give poor results.")
            self.T = T
            self.T_preamble = T_preamble

    @staticmethod
    def get_min_symbol_time(M: int, required_cycles: float, f0: int, fmax: int, minimal_sub_chirp_duration: bool = False):
        # Based on equation (3) from https://kirj.ee/public/Engineering/2011/issue_2/eng-2011-2-169-179.pdf
        # However, we make a slight adjustment, since we need to guarantee the min subchirp time. So we multiply that
        # by the number of subchirps.
        # It might be interesting to have separate subchirp length, each with at their specific min T
        if not minimal_sub_chirp_duration:
            f_subchirp_max = f0 + ((fmax - f0)/M)
            result = ((2*required_cycles) / (f0 + f_subchirp_max)) * M
        else:
            result = 0
            fdelta = (fmax - f0) / M
            for i in range(1, M+1):
                result += (2*required_cycles)/((2 * f0) + (((2 * i) - 1) * fdelta))

        return result

    def get_orthogonal_chirps(self):
        # The R matrix from step 4
        # R = [[4, 6, 2, 1, 5, 3],
        #      [3, 4, 6, 2, 1, 5],
        #      [5, 3, 4, 6, 2, 1],
        #      [1, 5, 3, 4, 6, 2],
        #      [2, 1, 5, 3, 4, 6],
        #      [6, 2, 1, 5, 3, 4]]
        #
        # The R matrix that reproduces figure 5
        if self.M == 6:
            R = [[3, 4, 1, 6, 2, 5],
                 [2, 3, 6, 5, 1, 4],
                 [1, 2, 5, 4, 6, 3],
                 [6, 1, 4, 3, 5, 2],
                 [5, 6, 3, 2, 4, 1],
                 [4, 5, 2, 1, 3, 6]]
        if self.M == 8:
            R = [
                [1, 8, 7, 4, 3, 5, 2, 6],
                [3, 6, 5, 2, 4, 7, 8, 1],
                [8, 5, 6, 7, 1, 2, 4, 3],
                [7, 1, 2, 5, 8, 6, 3, 4],
                [6, 7, 4, 3, 2, 1, 5, 8],
                [2, 4, 3, 6, 7, 8, 1, 5],
                [4, 2, 1, 8, 5, 3, 6, 7],
                [5, 3, 8, 1, 6, 4, 7, 2]
            ]
        # The following matrices where pre-generated with the matlab code
        elif self.M == 5:
            R = [[5, 1, 3, 4, 2],
                 [3, 5, 2, 1, 4],
                 [2, 3, 4, 5, 1],
                 [1, 4, 5, 2, 3],
                 [4, 2, 1, 3, 5]]
        elif self.M == 4:
            R = [[3, 2, 1, 4],
                 [2, 4, 3, 1],
                 [1, 3, 4, 2],
                 [4, 1, 2, 3]]
        elif self.M == 3:
            R = [[1, 2, 3],
                 [2, 3, 1],
                 [3, 1, 2]]
        elif self.M == 2:
            R = [[1, 2],
                 [2, 1]]
        elif self.M == 1:
            R = [[1], [2]]
        else:
            print(f"Incorrect M [{self.M}]")
            R = None

        symbols = []
        for r in R:
            chirp = []
            # We calculate the hybrid chirps here. We could also only use upward or downward chirps.
            for i, m in enumerate(r):
                if self.M != 1:
                    fsm = self.fs + ((m-1)*(self.fe - self.fs))/self.M
                    fem = fsm + (self.fe - self.fs)/self.M
                else:
                    fsm = self.fs
                    fem = self.fe

                if m % 2 == i % 2:
                    chirp.append((fem, fsm))
                else:
                    chirp.append((fsm, fem))

            symbols.append(chirp)

        return symbols[self.orthogonal_pair_offset:self.orthogonal_pair_offset+2]

    def get_preamble(self, flipped: bool = False) -> np.ndarray:
        # Must be outside of the regular frequencies
        # A single tuple means M=1
        preamble = [(self.preamble_start, self.preamble_end)]

        # We want the preamble to be just one chirp, does not need to be orthogonal
        preamble = self.convert_bit_to_chrirp([preamble], 0, M=1, T=self.T_preamble)

        # We may flip the symbol if we need it for convolution
        if flipped:
            preamble = np.conjugate(np.flip(preamble))

        return preamble

    def convert_bit_to_chrirp(self, symbols, bit, M: int = None, T: float = None, no_window: bool = False,
                              blank_space: bool = True, minimal_sub_chirp_duration: bool = False) -> np.ndarray:
        # Symbols is the list of symbols we have at our disposal
        # Bit may only be 1/0

        # A 6ms black time to reduce Inter symbol Interference due to multipath
        # sound (343m/s) travels 2 meters in 6ms, so this should be sufficient for our range
        blank = np.zeros(int(self.fsample * self.blank_space_time))

        # We want to support non-default configs for the preamble
        if M is None:
            M = self.M

        if T is None:
            T = self.T - self.blank_space_time
        if minimal_sub_chirp_duration is False:
            Tb = T / M
            t = np.linspace(0, Tb, int(np.ceil(Tb * self.fsample)))
        else:
            # These will be overwritten in the encode function
            Tb = 0
            t = []

        def encode(symbol, Tb, t):
            c = np.array([])
            for subchirp in symbol:

                if minimal_sub_chirp_duration:
                    Tb = self.get_min_symbol_time(1, self.required_number_of_cycles, subchirp[0], subchirp[1])
                    t = np.linspace(0, Tb, int(np.ceil(Tb * self.fsample)))

                # Maybe shape the chirp with a window function, to reduce the phase difference click
                if not no_window:
                    window_kaiser = np.kaiser(len(t), beta=4)
                else:
                    window_kaiser = 1
                subchirp_signal = window_kaiser * self.volume * chirp(t, subchirp[0], Tb, subchirp[1])
                c = np.append(c, subchirp_signal)

            # Add blank
            if blank_space:
                c = np.append(c, blank)

            return c

        if bit == 0:
            return encode(symbols[0], Tb, t)
        elif bit == 1:
            return encode(symbols[1], Tb, t)
        else:
            print("bit is not 1 or 0!")

    def get_chirps_from_bits(self, symbols, bits, no_window: bool = False) -> [np.ndarray]:
        # Symbols is the list of symbols we have at our disposal
        # Bit must be a list of 1s and 0s
        chirps = []

        print(f"Converting to data: {bits}")
        print(f"Available symbols: {symbols}")

        for bit in bits:
            chirps.append(self.convert_bit_to_chrirp(symbols, bit, minimal_sub_chirp_duration=self.minimal_sub_chirp_duration, no_window=no_window))
        return chirps

    def convert_data_to_sound(self, data: str, filename: str = "temp.wav", no_window: bool = None) -> (str, np.ndarray):
        # Currently doing hybrid symbols, I think the paper concluded that there where no performance differences
        symbols = self.get_orthogonal_chirps()

        if no_window is None:
            no_window = self.no_window

        print(f"raw data: {data}")
        bits_to_send = tobits(data)
        chirps = self.get_chirps_from_bits(symbols, bits_to_send, no_window=no_window)

        preamble = self.get_preamble()

        concat_samples = np.array(preamble)
        for sample in chirps:
            concat_samples = np.append(concat_samples, np.array(sample))
        concat_samples = concat_samples * np.iinfo(np.int16).max
        concat_samples = concat_samples.astype(np.int16)

        write(filename, self.fsample, concat_samples)

        return filename, concat_samples


if __name__ == '__main__':
    data_to_send = "Hello, World!"

    oc = OChirpEncode(T=None)
    file, data = oc.convert_data_to_sound(data_to_send)
