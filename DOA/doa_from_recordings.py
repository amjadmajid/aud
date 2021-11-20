import numpy as np
import soundfile as sf
from doa_estimator import doa_estimator
import matplotlib.pyplot as plt


def doa_from_chirp(recordings, chirp_set, Fs):
    length = 0.0476
    c = 343

    M = chirp_set.shape[0]
    num_mics = recordings.shape[0]
    correlations_chirps = np.zeros((M, chirp_set.shape[1] + recordings.shape[1] - 1))
    for m in range(M):
        correlations_chirps[m] = np.correlate(recordings[0], chirp_set[m], 'full')
    # Determine which chirp has been detected by evaluating which correlation has the highest peak
    chirp_detected = np.argmax(np.max(correlations_chirps, axis=1))

    correlations_recordings = np.zeros((num_mics, chirp_set.shape[1] + recordings.shape[1] - 1))
    for n in range(num_mics):
        correlations_recordings[n] = np.correlate(recordings[n], chirp_set[chirp_detected], 'full')
        plt.plot(correlations_recordings[n])
        plt.title(f'mic{n}')
        plt.show()
    index = np.argmax(correlations_recordings, axis=1)
    tdoa = np.zeros(num_mics * 2)
    for m in range(num_mics):
        tdoa[m] = (index[m] - index[(m - 1) % num_mics]) / Fs
    for m in range(num_mics):
        tdoa[num_mics + m] = (index[m] - index[(m - 3) % num_mics]) / Fs
    print(tdoa)
    doa = doa_estimator(tdoa, length, c, num_mics)
    return chirp_detected, doa


Fs = 44100
M = 6
num_mics = 6
chirp = np.zeros((M, len(sf.read('chirp1.wav')[0])))
chirp_rec = np.zeros((num_mics, len(sf.read('recordings/recording-01.wav')[0])))
for i in range(M):
    chirp[i], Fs = sf.read(f'chirp{i + 1}.wav')

for i in range(num_mics):
    chirp_rec[i], Fs = sf.read(f'recordings/recording-0{i+1}.wav')

det, angle = doa_from_chirp(chirp_rec, chirp, Fs)
print(f"Chirp {det+1} detected\nEstimated angle: {angle} degrees")
