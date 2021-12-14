import paho.mqtt.client as mqtt
from MQTT_AAC_player import on_connect
from MQTT_AAC_control import get_hostname
from configuration import Configuration, get_configuration_encoder
from OChirpDecode import OChirpDecode
import shutil
from pathlib import Path
import multiprocessing
import time
from MQTT_AAC_control import record


def on_message(client, userdata, msg):
    print(f"Got message [{msg.payload}]")

    msg.payload = msg.payload.decode('utf-8')

    print(f"Message details: {msg.topic}  {str(msg.payload)}")

    record(msg.payload)

    print("Finished recording sound")
    client.publish("rec_done", 1)
    print("========================\n\n")


if __name__ == '__main__':
    print("Starting MQTT AAC recorder")

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    # connect
    client.connect(get_hostname())

    print("Starting loop...")

    client.loop_start()
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break

    client.loop_stop()
    print("Bye!")
