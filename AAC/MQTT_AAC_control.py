import paho.mqtt.client as mqtt
from configuration import Configuration
import time

DISTANCE_CM = 25
NUM_REPEATS = 10
DATA_TO_SEND = "2022-AAC-test-string123!"  # 24 characters, 192 bits


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
            play_done = False

            tx = f"{i}_{c.name}_{iteration}_{DATA_TO_SEND}"
            client.publish("playrec", tx)

            print("Waiting for command to finish...")
            while not (rec_done and play_done):
                pass
            print("Done.")
            time.sleep(0.25)

    print("Finished all configurations! Stopping...")
    client.loop_stop()
