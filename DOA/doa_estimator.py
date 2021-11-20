import numpy as np


def doa_estimator(tdoa, l, c, M):
    """Estimates the DOA given a set of TDOA values, length between the
    microphones, speed of sound, and number of microphones."""
    alpha = 360 / M      # The angle between microphones
    theta = np.zeros(M)
    for i in range(M):
        angle_pair = np.degrees(np.arcsin(np.clip(tdoa[i] * c / l, -1, 1)))  # compute the DOA based on one microphone pair
        print(np.clip(tdoa[i] * c / l, -1, 1), angle_pair)
        index = i - 1
        if index == -1:
            index = M-1
        if tdoa[M + index] >= 0:
            angle_pair = 180 - angle_pair
        theta[i] = (angle_pair - (i-1) * alpha) % 360
    print(theta)
    return round(360 - angle_mean(theta), 3)


def angle_mean(angles):
    """Computes the average angle given an array of angles in degrees."""
    complex_sum = 0
    for angle in angles:
        complex_sum += np.exp(1j*np.radians(angle))
    return np.angle(complex_sum, deg=True) % 360
