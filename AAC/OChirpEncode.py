import numpy as np
from BitManipulation import frombits, tobits
from scipy.signal import chirp
from scipy.io.wavfile import write


class OChirpEncode:

    """
        OChirpEncode

        This class is used to encode data into orthogonal chirps. This is done with the following function:
        `convert_data_to_sound`, which receives a string. See the example in the bottom of the file for more info.

        The constructor can receive a ton of arguments. Most are not interesting. Some notable parameters are:
            - T
                This is the final symbol time we want. If this is None, then we calculate this based on
                `required_number_of_cycles` and `minimize_sub_chirp_duration`.

            - M
                This is the number of orthogonal chirps to construct. We have hardcoded the matrices, please at the
                appropriate matrix to `get_orthogonal_chirps` if you get an error.

            - fs
                This is the chirp starting frequency (NOT SUB-CHIRP)

            - fe
                This is the chirp stopping frequency (NOT SUB-CHIRP)

            - f_preamble_start
                This is the preamble starting frequency (not orthogonal!)

            - f_preamble_end
                This is the preamble stopping frequency (not orthogonal!)

            - blank_space_time
                The gap between two symbols (not applied on the preamble)

            - T_preamble
                Duration of the preamble

            - orthogonal_pair_offset
                We use 2 symbols to transmit our data, but we have more symbols to our exposure. Normally we just take
                the first 2 symbols  (top 2 rows of the matrix). With this function we can select witch rows we want by
                shifting the selection.

            - required_number_of_cycles
                How many cycles should a sub-chirp minimally chirp contain? Used to calculate T if T is `None`.
                Otherwise used for a warning if T is too small

            - minimize_sub_chirp_duration
                Optimize T by calculating required_number_of_cycles per sub-chirp, instead of overestimating it
                This causes us to have subchirps of different length, so might effect orthogonality

            - volume
                The volume of the chirp [0-1]

            - no_window
                Whether we want to window the sub-chirps or not

    """

    def __init__(self, fsample: int = 44100, T: float = None, M: int = 8, fs: int = 5500, fe: int = 9500,
                 f_preamble_start: int = 100, f_preamble_end: int = 5500, blank_space_time: float = 0.000,
                 T_preamble: float = 0.2, orthogonal_pair_offset: int = 0, required_number_of_cycles: int = 5,
                 minimize_sub_chirp_duration: bool = False, volume: float = 1, no_window: bool = False):

        self.fsample = fsample
        self.M = M
        self.fs = fs
        self.fe = fe
        self.preamble_start = f_preamble_start
        self.preamble_end = f_preamble_end
        self.blank_space_time = blank_space_time
        self.orthogonal_pair_offset = orthogonal_pair_offset
        self.required_number_of_cycles = required_number_of_cycles

        """
            minimal sub chirp duration optimizes the sub-chirps such that the length is minimal (guarantees exact 
            required_number_of_cycles). This way, we reduce the symbol time. However, we this could impact the 
            orthogonality.
        """
        self.minimal_sub_chirp_duration = minimize_sub_chirp_duration

        self.volume = volume
        self.no_window = no_window

        if T is None:
            self.T = self.get_min_symbol_time(M, required_number_of_cycles, fs, fe, minimize_sub_chirp_duration) \
                        + blank_space_time
            self.T_preamble = T_preamble
            print(f"Calculated minimum symbol time: {self.T*1000:.1f} ms")
        elif minimize_sub_chirp_duration is False:
            min_symbol_time = self.get_min_symbol_time(M, 1, fs, fe) + blank_space_time
            if T < min_symbol_time:
                print(f"WARNING: The given T [{T*1000:.1f} ms] is smaller than required for a single cycle [{min_symbol_time*1000:.1f} ms]!")
                print("This will give poor results.")
            self.T = T
            self.T_preamble = T_preamble
        else:
            print(f"Incorrect configuration! T:{T} and minimal_sub_chirp_duration:{minimize_sub_chirp_duration}. "
                  "Please choose.")

    @staticmethod
    def get_min_symbol_time(M: int, required_cycles: float, f0: int, fmax: int, minimal_sub_chirp_duration: bool = False) -> float:
        """
            Based on equation (3) from https://kirj.ee/public/Engineering/2011/issue_2/eng-2011-2-169-179.pdf.
            However, we make a slight adjustment, since we need to guarantee the min subchirp time. So we multiply that
            by the number of subchirps. See equation 2/3/4 from our paper.
        """
        if not minimal_sub_chirp_duration:
            f_subchirp_max = f0 + ((fmax - f0)/M)
            result = ((2*required_cycles) / (f0 + f_subchirp_max)) * M
        else:
            result = 0
            fdelta = (fmax - f0) / M
            for i in range(1, M+1):
                result += (2*required_cycles)/((2 * f0) + (((2 * i) - 1) * fdelta))

        return result

    def get_orthogonal_chirps(self) -> list[list]:
        """
            This function returns a list of symbols
            Every entry in this list describes the orthogonal frequencies with a tuple containing a starting and
            stopping frequency. For example:
                    Symbol 0                    Symbol 1
            [[(0, 1000), (1000, 2000)], [(1000, 2000), (0, 1000)]]
        """
        # The R matrix that reproduces figure 5
        if self.M == 6:
            R = [[3, 4, 1, 6, 2, 5],
                 [2, 3, 6, 5, 1, 4],
                 [1, 2, 5, 4, 6, 3],
                 [6, 1, 4, 3, 5, 2],
                 [5, 6, 3, 2, 4, 1],
                 [4, 5, 2, 1, 3, 6]]

        # The following matrices where pre-generated with the matlab code
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

        # If we want just a regular chirp
        elif self.M == 1:
            R = [[1], [2]]
        else:
            print(f"Incorrect M [{self.M}]")
            R = None

        symbols = []
        for row in R:
            chirp = []
            # We calculate the hybrid chirps here.
            # TODO: allow us to select linear chirps here as well.
            for i, m in enumerate(row):
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

        # Only return two chirps based on the offset
        return symbols[self.orthogonal_pair_offset:self.orthogonal_pair_offset+2]

    def get_preamble(self, flipped: bool = False) -> np.ndarray:
        """
            Manipulate the functions a bit to generate a regular chirp such that we can use this as our preamble.
            Returns the preamble signal
        """

        # A single tuple means M=1
        preamble = [(self.preamble_start, self.preamble_end)]

        # We want the preamble to be just one chirp, does not need to be orthogonal
        preamble = self.convert_bit_to_chrirp([preamble], 0, M=1, T=self.T_preamble)

        # We may flip the symbol if we need it for convolution
        if flipped:
            preamble = np.conjugate(np.flip(preamble))

        return preamble

    def convert_bit_to_chrirp(self, symbols: list, bit: int, M: int = None, T: float = None, no_window: bool = False,
                              blank_space: bool = True, minimal_sub_chirp_duration: bool = False) -> np.ndarray:
        """
            Convert a single bit to chirp signal that we can transmit.
            This is done based on the provided symbols (not the symbol, but the list of tuples generated from
                `get_orthogonal_chirps`) and based on the bit (0/1).
            We can configure numerous parameters here:
                - M: How many sub-chirps, only used for the preamble (M=1), otherwise just the default
                - T: How long the chirp should be. again, only used for the preamble
                - no_window: Do we want to generate the symbol with or without a window around the sub-chirps
                - blank_space: Do we want to have a blank space between symbols? (False is same as blank_space_time=0)
                - minimal_sub_chirp_duration: How do we want to calculate the T? This value is always false for the
                                                preamble, but may vary for the symbols.
        """

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

    def get_chirps_from_bits(self, symbols: list, bits: list, no_window: bool = False) -> [np.ndarray]:
        """
            Convert the list of bits to a signal based on the provided list of symbols (from `get_orthogonal_chirps`)
        """

        chirps = []

        print(f"Converting to data: {bits}")
        print(f"Available symbols: {symbols}")

        for bit in bits:
            chirps.append(self.convert_bit_to_chrirp(symbols, bit,
                                                     minimal_sub_chirp_duration=self.minimal_sub_chirp_duration,
                                                     no_window=no_window))
        return chirps

    def convert_data_to_sound(self, data: str, filename: str = "temp.wav", no_window: bool = None) -> (str, np.ndarray):
        """
            Convert the data (string) to a sound. By default also writes it to a file `temp.wav`.
            The no_window parameters propagates through the system for just this encode.
        """

        symbols = self.get_orthogonal_chirps()

        # Choose the init argument als default
        if no_window is None:
            no_window = self.no_window

        print(f"raw data: {data}")
        bits_to_send = tobits(data)
        chirps = self.get_chirps_from_bits(symbols, bits_to_send, no_window=no_window)

        preamble = self.get_preamble()

        concat_samples = np.array(preamble)
        for sample in chirps:
            concat_samples = np.append(concat_samples, np.array(sample))

        # Convert float to int16
        concat_samples = concat_samples * np.iinfo(np.int16).max
        concat_samples = concat_samples.astype(np.int16)

        # Write the filename
        if filename is not None:
            write(filename, self.fsample, concat_samples)

        return filename, concat_samples


if __name__ == '__main__':
    import sounddevice as sd
    data_to_send = "Hello, World!"

    oc = OChirpEncode(T=None)
    file, data = oc.convert_data_to_sound(data_to_send)
    sd.play(data, oc.fsample, blocking=True)
