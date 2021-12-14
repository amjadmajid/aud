import paho.mqtt.client as mqtt
from configuration import Configuration, get_configuration_encoder
import sounddevice as sd
import time
from MQTT_AAC_control import get_hostname
import numpy as np
from OChirpOldFunctions import add_wgn


def play(msg: str):
    # f"{i}_{c}_{iteration}_{DATA_TO_SEND}"
    split_msg = msg.split('_')
    configuration_name = split_msg[1]
    data = split_msg[3]
    distance_cm = int(split_msg[4])

    print(f"Going to transmit : [{data}] with configuration [{configuration_name}] at distance [{distance_cm}]")

    config = Configuration[configuration_name]
    encoder = get_configuration_encoder(config)

    print("Encoding message")
    filename, data = encoder.convert_data_to_sound(data)

    # Add some white noise at the beginning and end, to make sure the speaker is initialized and does not stop too early
    z = np.ones(11000).astype(np.int16)
    z = add_wgn(z, -100).astype(np.int16)
    data = np.concatenate([z, np.array(data), z], dtype=np.int16).astype(np.int16)
    print(data)
    print("Playing...")
    sd.play(data, samplerate=encoder.fsample, blocking=True)


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {str(rc)}")
    client.subscribe("playrec")
    print("Waiting for message...")


def on_message(client, userdata, msg):
    print(f"Got message [{msg.payload}]")

    msg.payload = msg.payload.decode('utf-8')

    print(f"Message details: {msg.topic}  {str(msg.payload)}")

    play(msg.payload)

    print("Finished playing sound")
    client.publish("play_done", 1)
    print("========================\n\n")


if __name__ == '__main__':
    print("Starting MQTT AAC player")

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    # connect
    client.connect(get_hostname())

    print("Starting loop...")

    client.loop_start()
    while True:
        try:
            pass
        except KeyboardInterrupt:
            break

    client.loop_stop()
    print("Bye!")
