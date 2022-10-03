#time in ms of a sample and the preamble length
sample_length = 100
preamble_length = 220

#absolute path to folder in which the recorded samples are stored
recordings = "N:\AUD_Data\Line_Of_Sight\chirp_train_chirp_0s024_0\chirp_train_chirp_0s024_0\Raw_recordings\\"

#absolute path to folder in which to store snippets of audio
storage = ""

#filename for manual testing of the split
filename = "rec_050cm_000_locH2-FS.wav"

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import random
  





def detect_leading_silence(sound, silence_threshold=-20.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


for f in os.listdir(recordings):

    sound = AudioSegment.from_file(recordings + f)
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)    
    trimmed_sound = sound[start_trim:]

    # print(start_trim)
    # print(end_trim)
    # print(duration)


    start_samples = start_trim + preamble_length

    #200 samples in a raw recording, so need to find 200 samples afterwards

    for i in range(200):
        # print(i*sample_length)
        sound_byte = sound[start_samples+ i*sample_length: start_samples+(1+ i)*sample_length]
        #TODO: add sanity checks

        #Create splits to test/train/val
        train = 0.70
        val = 0.85

        number = random.uniform(0,1)

        if len(sound_byte) < 100:
            print("length too small")
            print(f)
            print(i)
            break
        folder = ""
        if number < train:
            folder = "train\\"
        elif number < val:
            folder = "validation\\"
        else:
            folder = "test\\"

        storagePath = "N:\AUD_Data\sampled\\"+ folder+ f[:-4] + "-"+str(i)+".wav"
        sound_byte.export(storagePath, format="wav")

    # print("finished split")


