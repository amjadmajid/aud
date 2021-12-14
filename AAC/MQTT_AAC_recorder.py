import paho.mqtt.client as mqtt
from MQTT_AAC_player import on_connect
from MQTT_AAC_control import get_hostname
from configuration import Configuration, get_configuration_encoder
from OChirpDecode import OChirpDecode
import shutil
from pathlib import Path


def record(msg: str, decode: bool = False):
    # f"{i}_{c}_{iteration}_{DATA_TO_SEND}"
    split_msg = msg.split('_')
    configuration_name = split_msg[1]
    data = split_msg[3]

    print(f"Going to record and decode : [{data}] with configuration [{configuration_name}]")

    config = Configuration[configuration_name]
    encoder = get_configuration_encoder(config)
    decoder = OChirpDecode(encoder=encoder, original_data=data)

    ber = decoder.decode_live(plot=decode, do_not_process=decode)

    print(f"Done. BER = [{ber}]. Saving recording...")

    Path("./range-recordings/").mkdir(parents=True, exist_ok=True)
    shutil.move("microphone.wav", "./range-recordings/" + msg + '.wav')


def on_message(client, userdata, msg):
    print(f"Got message [{msg.payload}]")

    msg.payload = msg.payload.decode('utf-8')

    print(f"Message details: {msg.topic}  {str(msg.payload)}")

    record(msg.payload, True)

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
            pass
        except KeyboardInterrupt:
            break

    client.loop_stop()
    print("Bye!")
