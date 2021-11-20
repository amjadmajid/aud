import itertools
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io.wavfile import write



def is_orthogonal(set_of_permutations, new_permutation):
    """Checks if the given permutation of numbers is orthogonal to the
       given set of permutations."""
    for perm in set_of_permutations:
        for j in range(len(new_permutation)):
            if perm[j] == new_permutation[j]:
                return False
    return True


def plot_freq(signal_frequency, t_c, M, n, handle):
    """Plots the frequency contents of a signal. M is the number of sub-chirps,
       t_c is the sub-chirp duration, n is the number of the chirp that should be
       placed in the plot title. 'handle' refers to the matplotlib object in which
       the plot should be made."""
    time = np.linspace(0, t_c, len(signal_frequency))*1000  # time axis (in ms)
    for i in range(M):
        # Each sub-chirp is plotted separately on the same plot to avoid vertical lines
        # between the sub-chirps in the plot.
        indx = int(i*len(signal_frequency)/M)
        timeaxis = time[indx+1:indx+int(len(signal_frequency)/M-1)]
        signalaxis = signal_frequency[indx+1:indx+int(len(signal_frequency)/M-1)]/1000
        handle.plot(timeaxis, signalaxis, 'k')  # plot sub-chirp
    handle.set(title='Chirp ' + str(n))


def optimizer(signal_set):
    """Return an optimization parameter for the given signal set."""
    M = signal_set.shape[0]
    autocor_peaks_sidelobes = np.zeros((M, 1))
    crosscorrelations = np.zeros((M, M, len(np.correlate(signal_set[1], signal_set[1], 'full'))))
    crosscor_peaks = np.zeros((M, M - 1))

    # get cross-correlations between each of the signals
    for i in range(M):
        for j in range(M):
            if i >= j:
                crosscorrelations[i][j] = np.correlate(signal_set[i], signal_set[j], 'full')

    # for m in range(M):
    #     peak_index = np.argmax(crosscorrelations[m][m])
    #     autocor_without_mainlobe = np.concatenate((crosscorrelations[m][m][:peak_index - 150],
    #                                                crosscorrelations[m][m][peak_index + 150:]))
    #     autocor_peaks_sidelobes[m] = np.max(autocor_without_mainlobe)
    #     n = 0
    #     for k in range(M):
    #         if m > k:
    #             crosscor_peaks[m][n] = np.max(crosscorrelations[k][m])
    #             n += 1
    # c_max = np.amax(crosscor_peaks)
    # a_max = np.max(autocor_peaks_sidelobes)
    min_autocors = np.zeros(M)
    indx = 0
    for m in range(M):
        min_autocors[m] = max(abs(crosscorrelations[m][m]))
    max_crosscorrs = np.zeros(int(np.sum(np.linspace(1, M-1))))
    for m in range(M):
        for n in range(M):
            if m > n:
                max_crosscorrs[indx] = max(abs(crosscorrelations[m][n]))
                indx += 1
    crosscor_max = max(max_crosscorrs)
    autocor_min = min(min_autocors)
    return crosscor_max/autocor_min


def sequence_generator(m, max_num, alpha):
    R = [k for k in range(1, m + 1)]
    permutations = set(itertools.permutations(R))
    if m < 5:
        permutation_sets = set()
        # create first row
        for permutation in permutations:
            perm_frozenset = set()
            perm_frozenset.add(permutation)
            perm_frozenset = frozenset(perm_frozenset)
            permutation_sets.add(perm_frozenset)
        for i in range(m - 1):
            new_permutation_set = set()
            for permutation_set in permutation_sets:
                for permutation in permutations:
                    if is_orthogonal(permutation_set, permutation):
                        permutation_list = list(permutation_set)
                        permutation_list.append(permutation)
                        new_permutation_set.add(frozenset(permutation_list))
            permutation_sets = new_permutation_set
    else:
        permutation_sets = set()
        permutations_list = list(permutations)
        while len(permutation_sets) < max_num:
            newset = set()
            while len(newset) < M:
                ind = random.randint(0, len(permutations_list) - 1)
                newset.add(permutations_list[ind])
            newset_frozen = frozenset(newset)
            permutation_sets.add(newset_frozen)
    return permutation_sets


def chirp_generator(f_s, f_e, M, f_sampling, t_c):
    """Generates a set of M orthogonal chirps of duration t_c"""
    t_b = t_c / M
    # t_c_vector = np.linspace(1 / f_sampling, t_c, int(t_c * f_sampling))
    t_b_vector = np.linspace(1 / f_sampling, t_b, int(t_b * f_sampling))

    f_up = np.zeros((M, len(t_b_vector)))
    f_down = np.zeros((M, len(t_b_vector)))
    y_up = np.zeros((M, len(t_b_vector)))
    y_down = np.zeros((M, len(t_b_vector)))

    # Generate sub-chirps
    for m in range(M):
        f_sm = f_s + m * (f_e - f_s) / M
        f_em = f_sm + (f_e - f_s) / M
        f_up[m] = f_sm + (f_em - f_sm) * t_b_vector / t_b
        f_down[m] = f_em - (f_em - f_sm) * t_b_vector / t_b
        y_up[m] = np.sin(2 * math.pi * np.multiply(t_b_vector, f_up[m]))    # up-subchirps
        y_down[m] = np.sin(2 * math.pi * np.multiply(t_b_vector, f_up[m]))  # down-subchirps
    # Generate set of orthogonal sequences
    r = list(sequence_generator(M, 800, 1))
    sub_signal = np.zeros((len(r), M, M, len(t_b_vector)))
    signal_set = np.zeros((len(r), M, M * len(t_b_vector)))

    for s in range(len(r)):
        orthogonal_set = list(r[s])
        for j in range(M):
            sequence = orthogonal_set[j]
            for k in range(M):
                sub_signal[s][j][k] = y_up[sequence[k] - 1]
                sub_signal_i = sub_signal[s][j]
                signal_set[s][j] = np.reshape(sub_signal_i, (1, M * len(t_b_vector)))
    K = []

    # Compute optimization parameter (named K) for each of the signal sets
    for i in range(len(r)):
        K.append(optimizer(signal_set[i]))
        print(f'chirp set {i} checked...')
    K_min_index = np.argmin(K)
    final_signal_set = signal_set[K_min_index]
    frequency_signals = get_frequency(K_min_index, r, f_up, M)
    return final_signal_set, frequency_signals, K_min_index, r


def get_frequency(k_min_index, r, f_vector, M):
    subsignal_frequencies = np.zeros((M, M, f_vector.shape[1]))
    frequency_signal = np.zeros((M, M*f_vector.shape[1]))
    R = list(r[k_min_index])
    for m in range(M):
        for n in range(M):
            subsignal_frequencies[m][n] = f_vector[R[m][n]-1]
        frequency_signal[m] = np.reshape(subsignal_frequencies[m], M*(f_vector.shape[1]))
    return frequency_signal


M = 6
f_start = 2000
f_stop = 4000
Fs = 44100
t_c = 0.100

signal, freq_test, k_min, r = chirp_generator(f_start, f_stop, M, Fs, t_c)

# Save chirps
# Generate chirps padded with silence to play on a JBL speaker
for i in range(M):
    scaled = np.int16(signal[i] / np.max(np.abs(signal[i])) * 32767)
    write(f'chirp{i+1}.wav', Fs, scaled)
    zero_padding_start = np.zeros(int(0.1 * Fs))
    zero_padding_end = np.zeros(int(0.5 * Fs))
    chirp_ext = np.concatenate((zero_padding_start, signal[i], zero_padding_end))
    scaled = np.int16(chirp_ext * 32767)
    write(f'chirp{i + 1}_extended.wav', Fs, scaled)
fig, axs = plt.subplots(2, 3)
# Plot
for i in range(2):
    for j in range(3):
        plot_freq(freq_test[i*2 + j], t_c, M, i*2 + j + 1, axs[i, j])
for ax in axs.flat:
    ax.set(xlabel='Time [ms]', ylabel='Frequency [kHz]')
for ax in axs.flat:
    ax.label_outer()

plt.rc('figure', titlesize=14)
fig.suptitle(f'Set of {M} orthogonal chirp waveforms ({f_start / 1000}-{f_stop / 1000} kHz)')
fig.tight_layout()
fig.show()
fig.savefig('chirps.png', dpi=400)

