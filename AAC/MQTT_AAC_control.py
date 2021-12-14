import paho.mqtt.client as mqtt
from configuration import Configuration, get_configuration_encoder
import time
import numpy as np
from OChirpOldFunctions import add_wgn
import sounddevice as sd
from OChirpDecode import OChirpDecode
import shutil
from pathlib import Path
import pyaudio
from scipy.io.wavfile import write

DISTANCE_CM = 25
NUM_REPEATS = 10
DATA_TO_SEND = "Hello, World!"  # 13 characters, 104 bits


def play_done_callback(client, userdata, message):
    global play_done
    play_done = True
    print("Play done")


def rec_done_callback(client, userdata, message):
    global rec_done
    rec_done = True
    print("Rec done")


def get_hostname():
    return "192.168.1.26"


def play(msg: str):
    # f"{i}_{c}_{iteration}_{DATA_TO_SEND}"
    split_msg = msg.split('_')
    configuration_name = split_msg[1]
    data = split_msg[3]
    distance_cm = int(split_msg[4])

    print(f"Going to transmit : [{data}] with configuration [{configuration_name}] at distance [{distance_cm}]")

    config = Configuration[configuration_name]
    encoder = get_configuration_encoder(config)

    # decoder = OChirpDecode(original_data=data, encoder=encoder)

    filename, data = encoder.convert_data_to_sound(data)

    sd.play(data, encoder.fsample, blocking=False)

    # decoder.decode_live(plot=False, do_not_process=False)

    # make sure we finished playing (decoder should block though)
    # sd.wait()


def record(msg: str, do_not_decode: bool = False):
    # f"{i}_{c}_{iteration}_{DATA_TO_SEND}"
    split_msg = msg.split('_')
    configuration_name = split_msg[1]
    data = split_msg[3]
    distance_cm = int(split_msg[4])

    print(f"Going to record and decode : [{data}] with configuration [{configuration_name}] at distance [{distance_cm}]")

    config = Configuration[configuration_name]
    encoder = get_configuration_encoder(config)
    decoder = OChirpDecode(encoder=encoder, original_data=data)

    # Plotting does not work, not main thread
    # ber = decoder.decode_live(plot=False, do_not_process=False)

    p = pyaudio.PyAudio()
    CHUNK = int(len(decoder.original_data_bits) * encoder.T * 1.25)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=encoder.fsample, input=True, frames_per_buffer=CHUNK)
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

    ber = decoder.decode_data(data, plot=False)

    write("microphone.wav", encoder.fsample, data.astype(np.int16))

    print(f"Done. BER = [{ber}]. Saving recording...")

    Path("./data/range-recordings/").mkdir(parents=True, exist_ok=True)
    shutil.move("microphone.wav", "./data/range-recordings/" + msg + '.wav')


if __name__ == "__main__":
    client = mqtt.Client()

    client.connect(get_hostname())

    client.subscribe("rec_done")
    client.subscribe("play_done")

    client.message_callback_add("rec_done", rec_done_callback)
    client.message_callback_add("play_done", play_done_callback)

    client.loop_start()
    print(f"Starting range test at {DISTANCE_CM}cm with data: [{DATA_TO_SEND}]")
    for c in Configuration:
        i = c.value[0]
        for iteration in range(NUM_REPEATS):
            print(f"Testing configuration {i} : {c.name}")

            rec_done = False
            play_done = True

            tx = f"{i}_{c.name}_{iteration}_{DATA_TO_SEND}_{DISTANCE_CM}"
            client.publish("playrec", tx)

            time.sleep(1)
            play(tx)
            # record(tx)

            sd.wait()

            print("Waiting for command to finish...")
            while not (rec_done and play_done):
                pass

            print("Done.")
            time.sleep(0.25)
            break
        break

    print("Finished all configurations! Stopping...")
    client.loop_stop()
