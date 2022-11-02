#time in ms of a sample and the preamble length
sample_length = 100
preamble_length = 220

#absolute path to folder in which the recorded samples are stored
headDir= "E:\\fake\\"
FS_offgrid = "Ordered_files_off_grid_FS\\Ordered_files_off_grid_FS\\Obstructed_Top\\Line_of_Sight\\"
recordings = "chirp_train_chirp_0s024_"
numbers = range(0,1)
suffix = "\Raw_recordings\\"

recordingsList = []
for i in numbers:
    recordingsList.append(headDir +FS_offgrid+ recordings + str(i) + "\\" + suffix)

#absolute path to folder in which to store snippets of audio
storage = ""

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import random
  





def detect_leading_silence(sound, silence_threshold=-30.0, chunk_size=10):
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


print("check split")
for recordingsPlace in recordingsList:
    print("started splitting")
    for f in os.listdir(recordingsPlace):
        #small if to only include FS
        if True:
            sound = AudioSegment.from_file(recordingsPlace + f)
            start_trim = detect_leading_silence(sound)
            end_trim = detect_leading_silence(sound.reverse())

            # print(start_trim)
            # print(end_trim)
            # print(duration)


            start_samples = start_trim + preamble_length

            if start_trim > 5000 or start_trim < 800:
                print("Questionable start found")
                print(start_trim)
                print(recordingsPlace + f)
                continue

            #200 samples in a raw recording, so need to find 200 samples afterwards

            for i in range(200):
                # print(i*sample_length)
                sound_byte = sound[start_samples+ i*sample_length: start_samples+(1+ i)*sample_length]

                #Create splits to test/train/val
                train = 0.70
                val = 0.85

                number = random.uniform(0,1)

                if len(sound_byte) < 100:
                    # print("length too small")
                    print(recordingsPlace + f)
                    print(i)
                    print(start_samples)
                    print(start_trim)
                    break
                folder = ""
                if number < train:
                    folder = "train\\"
                elif number < val:
                    folder = "validation\\"
                else:
                    folder = "test\\"

                #sessionID required to prevent overwriting similar measurements on different day
                sessionID_index = recordingsPlace.find("Raw") -3


                storagePath = "E:\\sampled\\"+ folder+ f[:-4] + "-"+str(i)+"-Session" + recordingsPlace[sessionID_index:sessionID_index+1] + ".wav"
                print(storagePath)
                sound_byte.export(storagePath, format="wav")

            # print("finished split")
    print("ended splitting ")


