import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve


class Localizer:
    """Localizes a source based on a recording from a 6-mic circular array."""

    def __init__(self, chirp_set, fs=44100, distance_between_mics=0.0475, speed_of_sound=343):
        self.chirp_set = chirp_set
        self.fs = fs
        self.d = distance_between_mics
        self.c = speed_of_sound
        self.M = chirp_set.shape[0]
        self.correlations_with_chirp_set = None
        self.correlations_recordings = None

    def recording_to_tdoa(self, recordings, chirp_index):
        num_mics = recordings.shape[0]
        recordings = recordings / 32767

        # Correlate each recorded signal with one of the recordings
        correlations_recordings = np.zeros((num_mics, recordings.shape[1] + self.chirp_set.shape[1] - 1))
        for n in range(num_mics):
            correlations_recordings[n] = oaconvolve(recordings[n], self.chirp_set[chirp_index, ::-1], 'full')

        index = np.argmax(correlations_recordings, axis=1)
        for i in range(6):
            if np.count_nonzero(np.logical_and((-10 < index - index[i]), (index - index[i] < 10))) > 2:
                med = index[i]
                break
            if i == 5:
                med = int(np.median(index))

        index = np.argmax(correlations_recordings[:, med-10:med+10], axis=1)
        self.correlations_recordings = correlations_recordings
        tdoa = np.zeros((2, num_mics))
        for m in range(num_mics):
            tdoa[0, m] = (index[m] - index[(m - 1) % num_mics]) / self.fs
        for m in range(num_mics):
            tdoa[1, m] = (index[m] - index[(m-3) % num_mics]) / self.fs
        return tdoa

    def tdoa_to_doa(self, tdoa):
        """Estimates the DOA given real_angle set of TDOA values, length between the
        microphones, speed of sound, and number of microphones."""
        alpha = 360 / 6      # The angle between microphones
        theta = np.zeros(6)
        for i in range(6):
            angle_pair = np.degrees(np.arcsin(np.clip(tdoa[0, i] * self.c / self.d, -1, 1)))  # compute the DOA based on one microphone pair
            index = i - 1
            if index == -1:
                index = -1
            if tdoa[1, index] >= 0:
                angle_pair = 180 - angle_pair
            theta[i] = (angle_pair - (i-1) * alpha) % 360
        return round(360 - self.angle_mean(theta), 3)

    def estimate_doa(self, recordings, chirp_index):
        tdoa = self.recording_to_tdoa(recordings, chirp_index)
        if tdoa is not None:
            doa = self.tdoa_to_doa(tdoa)
            return chirp_index, doa
        else:
            return None, None

    @staticmethod
    def angle_mean(angles):
        """Computes the average angle given an array of angles in degrees."""
        complex_sum = 0
        for angle in angles:
            complex_sum += np.exp(1j * np.radians(angle))
        return np.angle(complex_sum, deg=True) % 360

