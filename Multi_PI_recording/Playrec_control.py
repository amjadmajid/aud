import paho.mqtt.client as mqtt
import time
import sounddevice as sd
from scipy.io.wavfile import read
import numpy as np
from matplotlib import pyplot as plt

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

def play_done_callback(client, userdata, message):
    global play_done
    play_done = True
    print("play_done")

def rec_done_callback(client, userdata, message):
    global rec_done
    rec_done = True
    print("rec_done")

def on_message(client, userdata, message):
    print(msg.topic+" "+str(msg.payload))

def Input_parsing(dist, direction, LoS, edist, edirection, location, top, test, duration):

    args = ""
    if test:
        args = "{} -t".format(args)

    args = "{} --distance {}".format(args, dist)
    args = "{} --direction {}".format(args, direction)

    if not LoS:
        args = "{} --LoS".format(args)
        args = "{} --edistance {}".format(args, edist)
        args = "{} --edirection {}".format(args, edirection)

    if top:    
        args = "{} --top".format(args)

    args = "{} --location {}".format(args, location)
    args = "{} --duration {}".format(args, duration)

    return args



# User inputs start

# (Geodesic) souce location
dist = 50 #cm
direction = 0

# Line-of-Sight state
LoS = True

# (Euclidian) source location (only send if LoS == False)

edist = 0 #cm
edirection = 0


# Meta lobation of recording
location = "H2-IC02"

# Top state (if ther's an obustructio inbetween the mics)
top = True

# Testing flag (set to True to run trough program without actually playing/recording
test = False

# Music_files
# File names
if test:
    M = 2
    chirp_types = ["0s024"]
else:
    M = 8
    chirp_types = ["0s024", "0s048"]

music_names = []
for j in range(len(chirp_types)):
    for i in range(M):
        music_names.append('chirp_train_chirp_{}_{}'.format(chirp_types[j],i))

music_names = ['baseline', 'baseline_fast', 'balanced', 'fast']
music_names = ['fast']

# Length of the music files (seconds)
if test:
    duration = 2
else:
    duration = 5  # 30

# User inputs end
msg = Input_parsing(dist, direction, LoS, edist, edirection, location, top, test, duration)


rec_init = False
play_init = False

rec_done = rec_init
play_done = play_init

#connect to mqtt
client = mqtt.Client()

client.connect("192.168.1.196")

client.subscribe("rec_done")
client.subscribe("play_done")

client.message_callback_add("rec_done", rec_done_callback)
client.message_callback_add("play_done", play_done_callback)

# loop
client.loop_start()
print("playrec settings \n{}\n".format(msg))
for i in range(len(music_names)):
    for _ in range(15):
        print(music_names[i])

        rec_done = False
        play_done = True

        tx_args = "{} --music {}".format(msg, music_names[i])
        client.publish("playrec", tx_args)

        music = '../AAC/sample_chirps/{}.wav'.format(music_names[i])
        fs, data = read(music)

        # plt.figure()
        # plt.plot(data)
        z = np.zeros(1*44100).astype(np.int16)
        # z = add_wgn(z, -20)
        data = np.append(z, data)
        data = np.append(data, z)
        # plt.plot(data)
        # plt.show()

        # Ensure the other side is recording
        sd.play(data, fs, blocking=True)

        while not (rec_done and play_done):
            pass

        print("")

print("done")
print("playrec settings \n{}\n".format(msg))
client.loop_stop()
